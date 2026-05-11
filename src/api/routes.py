"""
routes.py — All API endpoint handlers.

Dependency pattern: get_datastore() reads from request.app.state, which is
set by the lifespan in main.py. Using Request here avoids a circular import
between routes.py and main.py.
"""
from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from src.data.loader import DataStore
from src.api.schemas import (
    ExplainResponse,
    ForecastPoint,
    ForecastResponse,
    RecommendationResponse,
    SeriesItem,
    SeriesListResponse,
    ShapDriver,
    SimulateRequest,
    SimulateResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared dependency
# ---------------------------------------------------------------------------

def get_datastore(request: Request) -> DataStore:
    return request.app.state.datastore


DataStoreDep = Annotated[DataStore, Depends(get_datastore)]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_series(ds: DataStore, unique_id: str) -> None:
    """Raise 404 if unique_id is not in the loaded DataStore."""
    if unique_id not in ds.series_meta:
        raise HTTPException(
            status_code=404,
            detail=f"Series '{unique_id}' not found. "
                   f"Call GET /series for the full list.",
        )


# ---------------------------------------------------------------------------
# GET /series
# ---------------------------------------------------------------------------

@router.get("/series", response_model=SeriesListResponse)
def list_series(ds: DataStoreDep) -> SeriesListResponse:
    """List all 86 avocado market series with region and type metadata."""
    items = [
        SeriesItem(
            unique_id=uid,
            region=meta["region"],
            avocado_type=meta["avocado_type"],
        )
        for uid, meta in ds.series_meta.items()
    ]
    return SeriesListResponse(total=len(items), series=items)


# ---------------------------------------------------------------------------
# GET /forecast/{unique_id}
# ---------------------------------------------------------------------------

@router.get("/forecast/{unique_id}", response_model=ForecastResponse)
def get_forecast(unique_id: str, ds: DataStoreDep) -> ForecastResponse:
    """
    Return the 12-week price forecast for one series.

    Includes point forecasts from all four models (AutoETS, NHITS, AutoARIMA,
    SeasonalNaive) and AutoETS conformal prediction interval (80 % coverage).
    PI bounds are series-level constants derived from the CV residual distribution.
    """
    _require_series(ds, unique_id)

    series_forecast = ds.forecast_df[ds.forecast_df["unique_id"] == unique_id].copy()
    pi_row = ds.pi_df[ds.pi_df["unique_id"] == unique_id].iloc[0]
    lower_q: float = pi_row["lower_q"]
    upper_q: float = pi_row["upper_q"]

    points = [
        ForecastPoint(
            ds=row["ds"],
            auto_ets=row["AutoETS"],
            auto_ets_lower=row["AutoETS"] + lower_q,
            auto_ets_upper=row["AutoETS"] + upper_q,
            nhits=row["NHITS"],
            auto_arima=row["AutoARIMA"],
            seasonal_naive=row["SeasonalNaive"],
        )
        for _, row in series_forecast.iterrows()
    ]
    return ForecastResponse(unique_id=unique_id, points=points)


# ---------------------------------------------------------------------------
# GET /recommend/{unique_id}
# ---------------------------------------------------------------------------

@router.get("/recommend/{unique_id}", response_model=RecommendationResponse)
def get_recommendation(unique_id: str, ds: DataStoreDep) -> RecommendationResponse:
    """
    Return the revenue-maximising price recommendation for one series.

    Optimal price and revenue uplift are computed via grid search (±30 % bounds)
    using the LightGBM demand model trained in notebook 04.
    """
    _require_series(ds, unique_id)

    row = ds.recommendations_df[
        ds.recommendations_df["unique_id"] == unique_id
    ].iloc[0]

    return RecommendationResponse(
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


# ---------------------------------------------------------------------------
# GET /explain/{unique_id}
# ---------------------------------------------------------------------------

@router.get("/explain/{unique_id}", response_model=ExplainResponse)
def get_explanation(unique_id: str, ds: DataStoreDep) -> ExplainResponse:
    """
    Return the top-3 SHAP demand drivers for one series.

    SHAP values are computed on a 3 000-row stratified sample in notebook 06.
    Each driver shows feature name, value, SHAP magnitude, and demand direction.
    """
    _require_series(ds, unique_id)

    series_shap = ds.shap_drivers_df[
        ds.shap_drivers_df["unique_id"] == unique_id
    ].sort_values("driver_rank")

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
    return ExplainResponse(unique_id=unique_id, drivers=drivers)


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest, ds: DataStoreDep) -> SimulateResponse:
    """
    Run live LightGBM inference for a custom price on one series.

    The most recent feature row for the series is used as context; only
    AveragePrice is swapped out. All other features (lags, rolling stats,
    calendar, region encodings) remain at their last observed values.

    This answers: "If I set price to X today, what demand would the model predict?"
    """
    _require_series(ds, req.unique_id)

    ctx_row = ds.latest_ctx_df[
        ds.latest_ctx_df["unique_id"] == req.unique_id
    ].iloc[0].copy()

    current_price: float = float(ctx_row["AveragePrice"])
    current_volume: float = float(ctx_row["Total Volume"])

    # Swap in the requested price before inference.
    ctx_row["AveragePrice"] = req.price

    bundle = ds.model_bundle
    X = ctx_row[bundle["features"]].to_frame().T.astype(float)
    predicted_log_vol: float = float(bundle["model"].predict(X)[0])
    predicted_vol: float = math.exp(predicted_log_vol)

    return SimulateResponse(
        unique_id=req.unique_id,
        price=req.price,
        predicted_log_volume=predicted_log_vol,
        predicted_volume=predicted_vol,
        current_price=current_price,
        current_volume=current_volume,
    )
