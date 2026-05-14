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

from fastapi import FastAPI

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

from src.api.middleware import AccessLogMiddleware  # noqa: E402

app.add_middleware(AccessLogMiddleware)


# ---------------------------------------------------------------------------
# Register routers (imported here to avoid circular imports)
# ---------------------------------------------------------------------------

from src.api import routes  # noqa: E402

app.include_router(routes.router)
