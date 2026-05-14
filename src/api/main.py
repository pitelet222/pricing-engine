"""
main.py — FastAPI application entry point.

Start with:
    uvicorn src.api.main:app --reload

The DataStore is loaded once at startup via FastAPI's lifespan context manager
and stored in app.state. Routes retrieve it via get_datastore() in routes.py,
which uses the Request object — this avoids a circular import between main and routes.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.auth import verify_api_key
from src.api.logging import JsonFormatter
from src.api.middleware import AccessLogMiddleware
from src.config import settings
from src.data.loader import load_datastore

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
async def lifespan(app: FastAPI):
    _log.info("Loading DataStore from %s ...", settings.outputs_dir)
    app.state.datastore = load_datastore()
    _log.info(
        "DataStore ready — %d series loaded.",
        len(app.state.datastore.series_meta),
    )
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
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------

from src.api import routes as _routes  # noqa: E402 — must come after app creation


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
# Protected business endpoints — auth applied as a router-level dependency
# ---------------------------------------------------------------------------

app.include_router(
    _routes.router,
    dependencies=[Depends(verify_api_key)],
)
