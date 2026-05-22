"""
GET  /v1/recommend/{unique_id}  — revenue-maximising price for one series.
POST /v1/batch-recommend        — recommendations for multiple series in one call.
"""
from __future__ import annotations

from typing import cast

from fastapi import APIRouter

from src.api.routes._deps import (
    _CACHE,
    DataStoreDep,
    _cache_hit,
    _cache_miss,
    _require_series,
)
from src.api.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    RecommendationResponse,
)
from src.data.loader import DataStore

router = APIRouter()


def _build_recommendation(ds: DataStore, unique_id: str) -> RecommendationResponse:
    """Build (or return cached) RecommendationResponse for one series."""
    _ep = "/recommend/{unique_id}"
    cache_key = f"recommend:{unique_id}"

    if cache_key in _CACHE:
        _cache_hit(_ep)
        return cast(RecommendationResponse, _CACHE[cache_key])

    _cache_miss(_ep)
    row = ds.recommendations_df[
        ds.recommendations_df["unique_id"] == unique_id
    ].iloc[0]

    result = RecommendationResponse(
        unique_id=unique_id,
        is_organic=bool(row["is_organic"]),
        current_price=row["current_price"],
        optimal_price=row["optimal_price"],
        price_change_pct=row["price_change_pct"],
        current_revenue=row["current_revenue"],
        optimal_revenue=row["optimal_revenue"],
        revenue_change_pct=row["revenue_change_pct"],
        elasticity=row["elasticity"],
    )
    _CACHE[cache_key] = result
    return result


@router.get("/recommend/{unique_id}", response_model=RecommendationResponse)
def get_recommendation(unique_id: str, ds: DataStoreDep) -> RecommendationResponse:
    """
    Return the revenue-maximising price recommendation for one series.

    Optimal price and revenue uplift are computed via grid search (±30% bounds)
    using the LightGBM demand model trained in notebook 04.
    """
    _require_series(ds, unique_id)
    return _build_recommendation(ds, unique_id)


@router.post("/batch-recommend", response_model=BatchRecommendResponse)
def batch_recommend(req: BatchRecommendRequest, ds: DataStoreDep) -> BatchRecommendResponse:
    """
    Retrieve revenue-maximising price recommendations for multiple series in one call.

    Unknown unique_ids are collected in not_found rather than raising a 404 —
    the caller receives recommendations for all valid IDs and a list of which
    IDs were unrecognised. This is more useful than an all-or-nothing failure
    when querying a large portfolio.

    All results are served from the same in-process cache as GET /recommend/{id},
    so repeated batch calls for the same series cost nothing after the first.
    """
    results: list[RecommendationResponse] = []
    not_found: list[str] = []

    for uid in req.unique_ids:
        if uid not in ds.series_meta:
            not_found.append(uid)
        else:
            results.append(_build_recommendation(ds, uid))

    return BatchRecommendResponse(
        requested=len(req.unique_ids),
        found=len(results),
        not_found=not_found,
        results=results,
    )
