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
import logging.config
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.auth import verify_api_key
from src.api.middleware import AccessLogMiddleware
from src.config import settings
from src.data.loader import load_datastore

# ---------------------------------------------------------------------------
# Logging — configured before anything else so startup messages are captured.
# ---------------------------------------------------------------------------

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["console"],
    },
})

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
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(AccessLogMiddleware)


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------

from src.api import routes as _routes  # noqa: E402

# /health is already defined in routes.py with tags=["meta"].
# /metrics is mounted here so Prometheus scrapers always reach it.

@app.get(
    "/metrics",
    tags=["meta"],
    include_in_schema=False,  # keep it out of the OpenAPI user docs
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
