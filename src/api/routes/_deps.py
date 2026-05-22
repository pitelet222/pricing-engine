"""
_deps.py — Shared dependencies for all route sub-modules.

Contains the module-level response cache, the DataStore dependency, and the
small helpers (_require_series, _cache_hit, _cache_miss) used by every endpoint.

Kept in a separate module so sub-modules can import from here without creating
circular imports with __init__.py.
"""
from __future__ import annotations

from typing import Annotated, Any, cast

from fastapi import Depends, HTTPException, Request

from src.api.metrics import CACHE_HITS, CACHE_MISSES
from src.data.loader import DataStore

# Module-level response cache — keyed by "<endpoint>:<unique_id>" or "<endpoint>".
# Never invalidated during a process's lifetime; restart the server to refresh.
_CACHE: dict[str, Any] = {}


def get_datastore(request: Request) -> DataStore:
    return cast(DataStore, request.app.state.datastore)


DataStoreDep = Annotated[DataStore, Depends(get_datastore)]


def _cache_hit(endpoint: str) -> None:
    CACHE_HITS.labels(endpoint=endpoint).inc()


def _cache_miss(endpoint: str) -> None:
    CACHE_MISSES.labels(endpoint=endpoint).inc()


def _require_series(ds: DataStore, unique_id: str) -> None:
    """Raise 404 if unique_id is not in the loaded DataStore."""
    if unique_id not in ds.series_meta:
        raise HTTPException(
            status_code=404,
            detail=f"Series '{unique_id}' not found. "
                   f"Call GET /series for the full list.",
        )
