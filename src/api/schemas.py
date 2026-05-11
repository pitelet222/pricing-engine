"""
schemas.py — Pydantic request / response models for every API endpoint.

Field names and types are derived directly from the artifact columns in
data/outputs/. If a notebook changes a column name, update here first.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# GET /series
# ---------------------------------------------------------------------------

class SeriesItem(BaseModel):
    unique_id: str
    region: str
    avocado_type: Literal["organic", "conventional"]


class SeriesListResponse(BaseModel):
    total: int
    series: list[SeriesItem]


# ---------------------------------------------------------------------------
# GET /forecast/{unique_id}
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    ds: date
    auto_ets: float = Field(description="AutoETS point forecast (USD)")
    auto_ets_lower: float = Field(
        description="AutoETS + conformal lower quantile (80 % PI)"
    )
    auto_ets_upper: float = Field(
        description="AutoETS + conformal upper quantile (80 % PI)"
    )
    nhits: float
    auto_arima: float
    seasonal_naive: float


class ForecastResponse(BaseModel):
    unique_id: str
    pi_coverage: float = Field(
        default=0.80,
        description="Nominal conformal prediction-interval coverage",
    )
    points: list[ForecastPoint]


# ---------------------------------------------------------------------------
# GET /recommend/{unique_id}
# ---------------------------------------------------------------------------

class RecommendationResponse(BaseModel):
    unique_id: str
    is_organic: bool
    current_price: float = Field(description="Most recent actual price (USD)")
    optimal_price: float = Field(description="Revenue-maximising price (USD)")
    price_change_pct: float = Field(
        description="(optimal - current) / current × 100"
    )
    current_revenue: float
    optimal_revenue: float
    revenue_change_pct: float = Field(
        description="(optimal_revenue - current_revenue) / current_revenue × 100"
    )
    elasticity: float = Field(
        description="Point price elasticity of demand (∂log_vol / ∂log_price)"
    )


# ---------------------------------------------------------------------------
# GET /explain/{unique_id}
# ---------------------------------------------------------------------------

class ShapDriver(BaseModel):
    driver_rank: int = Field(ge=1, le=3)
    feature: str
    shap_value: float
    abs_shap_value: float
    direction: Literal["increases_demand", "decreases_demand"]
    feature_value: float


class ExplainResponse(BaseModel):
    unique_id: str
    drivers: list[ShapDriver]


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    unique_id: str
    price: float = Field(gt=0, description="Custom price to test (USD per avocado)")


class SimulateResponse(BaseModel):
    unique_id: str
    price: float
    predicted_log_volume: float
    predicted_volume: float = Field(description="exp(predicted_log_volume)")
    current_price: float = Field(
        description="Most recent actual price for this series"
    )
    current_volume: float = Field(
        description="Most recent actual total volume for this series"
    )
