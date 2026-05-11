"""
main.py — FastAPI application entry point.

Start with:
    uvicorn src.api.main:app --reload

The DataStore is loaded once at startup via FastAPI's lifespan context manager
and stored in app.state. Routes retrieve it via get_datastore() in routes.py,
which uses the Request object — this avoids a circular import between main and routes.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.data.loader import load_datastore


# ---------------------------------------------------------------------------
# Lifespan — load all artefacts once, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.datastore = load_datastore()
    yield
    # Nothing to release (in-memory DataFrames); placeholder for future cleanup.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Avocado Pricing Engine",
    description=(
        "Dynamic pricing recommendations, demand forecasts, uncertainty "
        "quantification, and SHAP explainability for 86 avocado market series."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Register routers (imported here to avoid circular imports)
# ---------------------------------------------------------------------------

from src.api import routes  # noqa: E402  — must come after app is defined

app.include_router(routes.router)
