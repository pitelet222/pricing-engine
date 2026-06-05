"""
main.py — FastAPI application entry point.

Start with:
    uvicorn src.api.main:app --reload

The DataStore is loaded once at startup via FastAPI's lifespan context manager
and stored in app.state. Routes retrieve it via get_datastore() in routes/_deps.py,
which uses the Request object — this avoids a circular import between main and routes.
"""

from __future__ import annotations

import json as _json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.auth import verify_api_key
from src.api.logging import JsonFormatter
from src.api.middleware import AccessLogMiddleware
from src.api.schemas import HealthResponse
from src.config import settings
from src.data.loader import load_datastore
from src.data.manifest import check_manifest

# ---------------------------------------------------------------------------
# Logging — configured before anything else so startup messages are captured.
#
# "text" (default): human-readable lines for local development.
# "json"           : structured JSON lines for ELK / Grafana Loki ingestion.
#                    Activate with PRICING_LOG_FORMAT=json.
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    if settings.log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers = [handler]


_configure_logging()

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load all artefacts once, release on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _log.info("Loading DataStore from %s ...", settings.outputs_dir)
    app.state.datastore = load_datastore()
    _log.info(
        "DataStore ready — %d series loaded.",
        len(app.state.datastore.series_meta),
    )

    # Artifact checksum verification — results exposed on GET /health.
    manifest_issues = check_manifest(settings.outputs_dir, settings.processed_dir)
    for issue in manifest_issues:
        _log.warning("manifest: %s", issue)
    app.state.manifest_ok = len(manifest_issues) == 0

    # Artifact generation timestamp from manifest.json.
    app.state.artifacts_generated_at = None
    manifest_path = settings.outputs_dir / "manifest.json"
    if manifest_path.exists():
        try:
            payload = _json.loads(manifest_path.read_text(encoding="utf-8"))
            app.state.artifacts_generated_at = payload.get("generated_at")
        except Exception:
            pass

    yield
    _log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# Authentication is disabled by default (PRICING_API_KEY not set).
# Set PRICING_API_KEY=<secret> in production to require X-API-Key on all
# business endpoints. /health and /metrics are always public.
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    # /v1/ prefix is applied to all business routes via include_router below.
    # Infrastructure endpoints (/health, /metrics) are registered directly on app.
)


# ---------------------------------------------------------------------------
# Middleware  (registered in reverse execution order — last added = outermost)
#
#   Request flow:  CORS → GZip → AccessLog → handler
#   Response flow: handler → AccessLog → GZip → CORS
#
# CORS is outermost so OPTIONS preflight is handled before auth/business logic.
# GZip compresses the full response before CORS headers are attached.
# AccessLog captures precise wall-clock timing closest to the handler.
# ---------------------------------------------------------------------------

app.add_middleware(AccessLogMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ---------------------------------------------------------------------------
# Public endpoints — no auth, no version prefix
# ---------------------------------------------------------------------------

from src.api.routes import router as _v1_router  # noqa: E402 — must come after app creation


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    """
    Liveness / readiness check. Always public — no API key required.

    Returns HTTP 200 as long as the DataStore loaded successfully.
    `manifest_ok` and `artifacts_generated_at` reflect the artifact state
    recorded at startup; they do not re-read the filesystem on each call.
    Suitable for Docker and Kubernetes health probes.
    """
    ds = request.app.state.datastore
    return HealthResponse(
        status="ok",
        version=settings.api_version,
        series_loaded=len(ds.series_meta),
        manifest_ok=request.app.state.manifest_ok,
        artifacts_generated_at=request.app.state.artifacts_generated_at,
    )


@app.get(
    "/metrics",
    tags=["meta"],
    include_in_schema=False,
    summary="Prometheus metrics",
)
def prometheus_metrics() -> Response:
    """Prometheus text exposition endpoint. Scraped by Prometheus server."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Protected business endpoints — auth + /v1/ prefix applied at router level
# ---------------------------------------------------------------------------

app.include_router(
    _v1_router,
    prefix="/v1",
    dependencies=[Depends(verify_api_key)],
)
