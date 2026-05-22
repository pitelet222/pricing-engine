"""
src/api/routes/ — FastAPI route sub-modules assembled into a single router.

Each sub-module owns one logical group of endpoints:
  series.py   — GET /series
  forecast.py — GET /forecast/{unique_id}
  pricing.py  — GET /recommend/{unique_id}, POST /batch-recommend
  explain.py  — GET /explain/{unique_id}, GET /uncertainty/{unique_id}
  simulate.py — POST /simulate

Shared state (response cache, DataStore dependency, helpers) lives in _deps.py
so sub-modules can import from there without circular imports.

The combined `router` is imported by main.py and mounted at /v1/.
`_CACHE` is re-exported here so existing references (e.g. test fixtures that
call `src.api.routes._CACHE.clear()`) continue to work without change.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.routes._deps import _CACHE as _CACHE  # re-export for backward compat
from src.api.routes.explain import router as _explain_router
from src.api.routes.forecast import router as _forecast_router
from src.api.routes.pricing import router as _pricing_router
from src.api.routes.series import router as _series_router
from src.api.routes.simulate import router as _simulate_router

router = APIRouter()
router.include_router(_series_router)
router.include_router(_forecast_router)
router.include_router(_pricing_router)
router.include_router(_explain_router)
router.include_router(_simulate_router)
