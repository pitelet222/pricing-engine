"""GET /v1/forecast/{unique_id} — 12-week price forecast for one series."""
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
from src.api.schemas import ForecastPoint, ForecastResponse

router = APIRouter()


@router.get("/forecast/{unique_id}", response_model=ForecastResponse)
def get_forecast(unique_id: str, ds: DataStoreDep) -> ForecastResponse:
    """
    Return the 12-week price forecast for one series.

    Includes point forecasts from all ensemble components and the
    Ensemble_weighted conformal PI (90% nominal coverage).
    PI bounds are series-level constants derived from the CV residual distribution.
    """
    _require_series(ds, unique_id)
    _ep = "/forecast/{unique_id}"
    cache_key = f"forecast:{unique_id}"

    if cache_key in _CACHE:
        _cache_hit(_ep)
        return cast(ForecastResponse, _CACHE[cache_key])

    _cache_miss(_ep)
    series_forecast = ds.forecast_df[ds.forecast_df["unique_id"] == unique_id].copy()
    pi_row = ds.pi_df[ds.pi_df["unique_id"] == unique_id].iloc[0]
    lower_q: float = pi_row["lower_q"]
    upper_q: float = pi_row["upper_q"]

    points = [
        ForecastPoint(
            ds=row["ds"],
            ensemble_weighted=row["Ensemble_weighted"],
            ensemble_lower=row["Ensemble_weighted"] + lower_q,
            ensemble_upper=row["Ensemble_weighted"] + upper_q,
            mstl_ets=row["MSTL_ETS"],
            mstl_arima=row["MSTL_ARIMA"],
            mstl_theta=row["MSTL_Theta"],
            nhits=row["NHITS"],
            nbeatsx=row.get("NBEATSx"),
            dlinear=row.get("DLinear"),
            seasonal_naive=row["SeasonalNaive"],
        )
        for _, row in series_forecast.iterrows()
    ]
    result = ForecastResponse(unique_id=unique_id, points=points)
    _CACHE[cache_key] = result
    return result
