"""GET /v1/series — list all 86 avocado market series."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.routes._deps import (
    _CACHE,
    DataStoreDep,
    _cache_hit,
    _cache_miss,
)
from src.api.schemas import SeriesItem, SeriesListResponse

router = APIRouter()


@router.get("/series", response_model=SeriesListResponse)
def list_series(
    ds: DataStoreDep,
    limit: int = Query(default=86, ge=1, le=86, description="Max series to return (1–86)"),
    offset: int = Query(default=0, ge=0, description="Starting index"),
) -> SeriesListResponse:
    """
    List avocado market series with region and type metadata.

    Supports optional pagination via `limit` and `offset`. The `total` field
    always reflects the full 86-series count regardless of page parameters.
    """
    _ep = "/series"
    # Cache the full ordered list; slice per request (O(limit), zero I/O).
    _all_key = f"{_ep}:all"
    if _all_key not in _CACHE:
        _cache_miss(_ep)
        all_items: list[SeriesItem] = [
            SeriesItem(
                unique_id=uid,
                region=meta["region"],
                avocado_type=meta["avocado_type"],
            )
            for uid, meta in ds.series_meta.items()
        ]
        _CACHE[_all_key] = all_items
    else:
        _cache_hit(_ep)
        all_items = _CACHE[_all_key]
    page = all_items[offset : offset + limit]
    return SeriesListResponse(total=len(all_items), limit=limit, offset=offset, series=page)
