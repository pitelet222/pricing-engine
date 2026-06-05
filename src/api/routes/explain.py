"""
GET /v1/explain/{unique_id}     — top-3 SHAP demand drivers for one series.
GET /v1/uncertainty/{unique_id} — three-strategy uncertainty analysis for one series.
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
    ExplainResponse,
    PricingStrategy,
    ShapDriver,
    UncertaintyResponse,
)

router = APIRouter()


@router.get("/explain/{unique_id}", response_model=ExplainResponse)
def get_explanation(unique_id: str, ds: DataStoreDep) -> ExplainResponse:
    """
    Return the top-3 SHAP demand drivers for one series.

    SHAP values are computed on a 3 000-row stratified sample in notebook 06.
    Each driver shows feature name, value, SHAP magnitude, and demand direction.
    """
    _require_series(ds, unique_id)
    _ep = "/explain/{unique_id}"
    cache_key = f"explain:{unique_id}"

    if cache_key in _CACHE:
        _cache_hit(_ep)
        return cast(ExplainResponse, _CACHE[cache_key])

    _cache_miss(_ep)
    series_shap = ds.shap_drivers_df[ds.shap_drivers_df["unique_id"] == unique_id].sort_values(
        "driver_rank"
    )

    drivers = [
        ShapDriver(
            driver_rank=int(row["driver_rank"]),
            feature=row["feature"],
            shap_value=row["shap_value"],
            abs_shap_value=row["abs_shap_value"],
            direction=row["direction"],
            feature_value=row["feature_value"],
        )
        for _, row in series_shap.iterrows()
    ]
    result = ExplainResponse(unique_id=unique_id, drivers=drivers)
    _CACHE[cache_key] = result
    return result


@router.get("/uncertainty/{unique_id}", response_model=UncertaintyResponse)
def get_uncertainty(unique_id: str, ds: DataStoreDep) -> UncertaintyResponse:
    """
    Return the three-strategy uncertainty analysis for one series.

    Conservative, balanced, and aggressive strategies each offer a different
    risk/return trade-off. Risk metrics (downside_risk_pct, uplift_sharpe,
    strategies_agree) help calibrate how reliable the recommendations are.
    All values are derived from the conformal prediction intervals computed
    in notebook 05.
    """
    _require_series(ds, unique_id)
    _ep = "/uncertainty/{unique_id}"
    cache_key = f"uncertainty:{unique_id}"

    if cache_key in _CACHE:
        _cache_hit(_ep)
        return cast(UncertaintyResponse, _CACHE[cache_key])

    _cache_miss(_ep)
    row = ds.uncertainty_df[ds.uncertainty_df["unique_id"] == unique_id].iloc[0]

    result = UncertaintyResponse(
        unique_id=unique_id,
        is_organic=bool(row["is_organic"]),
        current_price=float(row["current_price"]),
        conservative=PricingStrategy(
            opt_price=float(row["opt_price_conservative"]),
            rev_uplift_pct=float(row["rev_uplift_conservative"]),
        ),
        balanced=PricingStrategy(
            opt_price=float(row["opt_price_balanced"]),
            rev_uplift_pct=float(row["rev_uplift_balanced"]),
        ),
        aggressive=PricingStrategy(
            opt_price=float(row["opt_price_aggressive"]),
            rev_uplift_pct=float(row["rev_uplift_aggressive"]),
        ),
        rev_p10=float(row["rev_p10_current"]),
        rev_p50=float(row["rev_p50_current"]),
        rev_p90=float(row["rev_p90_current"]),
        rev_spread_pct=float(row["rev_spread_pct"]),
        downside_risk_pct=float(row["downside_risk_pct"]),
        uplift_mean=float(row["uplift_mean"]),
        uplift_std=float(row["uplift_std"]),
        uplift_sharpe=float(row["uplift_sharpe"]),
        strategies_agree=bool(row["strategies_agree"]),
        price_direction_conservative=int(row["price_direction_conservative"]),
        price_direction_aggressive=int(row["price_direction_aggressive"]),
    )
    _CACHE[cache_key] = result
    return result
