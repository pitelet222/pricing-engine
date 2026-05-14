"""
schemas.py — Pydantic request / response models for every API endpoint.

Field names and types are derived directly from the artifact columns in
data/outputs/. If a notebook changes a column name, update here first.

OpenAPI examples use real Albany_conventional data so the /docs page works
as a live demo without any additional setup.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    series_loaded: int = Field(description="Number of market series in the loaded DataStore")
    version: str = Field(description="API version string")

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "ok", "series_loaded": 86, "version": "0.1.0"}]
        }
    }


# ---------------------------------------------------------------------------
# GET /series
# ---------------------------------------------------------------------------

class SeriesItem(BaseModel):
    unique_id: str
    region: str
    avocado_type: Literal["organic", "conventional"]


class SeriesListResponse(BaseModel):
    total: int = Field(description="Total number of series in the DataStore (always 86)")
    limit: int = Field(description="Maximum items returned in this response")
    offset: int = Field(description="Starting index of this response")
    series: list[SeriesItem]


# ---------------------------------------------------------------------------
# GET /forecast/{unique_id}
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    ds: date
    ensemble_weighted: float = Field(description="Ensemble (inverse-MAE weighted) point forecast (USD)")
    ensemble_lower: float = Field(
        description="Ensemble_weighted + conformal lower quantile (90 % PI)"
    )
    ensemble_upper: float = Field(
        description="Ensemble_weighted + conformal upper quantile (90 % PI)"
    )
    mstl_ets: float = Field(description="MSTL + AutoETS component model forecast")
    mstl_arima: float = Field(description="MSTL + AutoARIMA component model forecast")
    mstl_theta: float = Field(description="MSTL + AutoTheta component model forecast")
    nhits: float = Field(description="NHITS neural forecast")
    nbeatsx: float | None = Field(default=None, description="NBEATSx neural forecast (None if not computed)")
    dlinear: float | None = Field(default=None, description="DLinear neural forecast (None if not computed)")
    seasonal_naive: float = Field(description="SeasonalNaive baseline forecast")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ds": "2018-04-01",
                "ensemble_weighted": 1.392,
                "ensemble_lower": 0.990,
                "ensemble_upper": 1.558,
                "mstl_ets": 1.306,
                "mstl_arima": 1.306,
                "mstl_theta": 1.316,
                "nhits": 1.506,
                "nbeatsx": None,
                "dlinear": None,
                "seasonal_naive": 1.62,
            }]
        }
    }


class ForecastResponse(BaseModel):
    unique_id: str
    pi_coverage: float = Field(
        default=0.90,
        description="Nominal conformal prediction-interval coverage",
    )
    points: list[ForecastPoint]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "unique_id": "Albany_conventional",
                "pi_coverage": 0.90,
                "points": ["... 12 ForecastPoint objects ..."],
            }]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "unique_id": "Albany_conventional",
                "is_organic": False,
                "current_price": 1.57,
                "optimal_price": 1.76,
                "price_change_pct": 12.0,
                "current_revenue": 136376.88,
                "optimal_revenue": 147733.40,
                "revenue_change_pct": 8.33,
                "elasticity": -0.054,
            }]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "driver_rank": 1,
                "feature": "is_organic",
                "shap_value": 1.530,
                "abs_shap_value": 1.530,
                "direction": "increases_demand",
                "feature_value": 0.0,
            }]
        }
    }


class ExplainResponse(BaseModel):
    unique_id: str
    drivers: list[ShapDriver]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "unique_id": "Albany_conventional",
                "drivers": [
                    {"driver_rank": 1, "feature": "is_organic", "shap_value": 1.530,
                     "abs_shap_value": 1.530, "direction": "increases_demand", "feature_value": 0.0},
                    {"driver_rank": 2, "feature": "region_volume_level", "shap_value": -1.083,
                     "abs_shap_value": 1.083, "direction": "decreases_demand", "feature_value": 10.742},
                    {"driver_rank": 3, "feature": "plu_share_4770", "shap_value": 0.207,
                     "abs_shap_value": 0.207, "direction": "increases_demand", "feature_value": 0.000438},
                ],
            }]
        }
    }


# ---------------------------------------------------------------------------
# GET /uncertainty/{unique_id}
# ---------------------------------------------------------------------------

class PricingStrategy(BaseModel):
    opt_price: float = Field(description="Revenue-optimising price for this strategy (USD)")
    rev_uplift_pct: float = Field(
        description="Expected revenue uplift vs current price (%)"
    )


class UncertaintyResponse(BaseModel):
    unique_id: str
    is_organic: bool
    current_price: float = Field(description="Most recent actual price (USD)")
    conservative: PricingStrategy = Field(
        description="Low-risk strategy: modest price adjustment"
    )
    balanced: PricingStrategy = Field(
        description="Balanced risk/return strategy"
    )
    aggressive: PricingStrategy = Field(
        description="High-return strategy: larger price move"
    )
    rev_p10: float = Field(description="10th-percentile revenue at current price (USD)")
    rev_p50: float = Field(description="Median revenue at current price (USD)")
    rev_p90: float = Field(description="90th-percentile revenue at current price (USD)")
    rev_spread_pct: float = Field(
        description="(rev_p90 − rev_p10) / rev_p50 × 100 — revenue uncertainty width"
    )
    downside_risk_pct: float = Field(
        description="Probability-weighted downside vs median revenue (%)"
    )
    uplift_mean: float = Field(description="Mean expected revenue uplift across price grid (%)")
    uplift_std: float = Field(description="Std dev of revenue uplift across price grid")
    uplift_sharpe: float = Field(description="uplift_mean / uplift_std — risk-adjusted score")
    strategies_agree: bool = Field(
        description="True when conservative and aggressive both recommend the same price direction"
    )
    price_direction_conservative: int = Field(
        description="Conservative strategy price direction: +1 = increase, −1 = decrease"
    )
    price_direction_aggressive: int = Field(
        description="Aggressive strategy price direction: +1 = increase, −1 = decrease"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "unique_id": "Albany_conventional",
                "is_organic": False,
                "current_price": 1.57,
                "conservative": {"opt_price": 1.65, "rev_uplift_pct": 4.2},
                "balanced": {"opt_price": 1.72, "rev_uplift_pct": 8.6},
                "aggressive": {"opt_price": 1.80, "rev_uplift_pct": 12.1},
                "rev_p10": 90_000.0,
                "rev_p50": 136_376.88,
                "rev_p90": 190_000.0,
                "rev_spread_pct": 73.5,
                "downside_risk_pct": 8.3,
                "uplift_mean": 8.2,
                "uplift_std": 3.1,
                "uplift_sharpe": 2.6,
                "strategies_agree": True,
                "price_direction_conservative": 1,
                "price_direction_aggressive": 1,
            }]
        }
    }


# ---------------------------------------------------------------------------
# POST /batch-recommend
# ---------------------------------------------------------------------------

class BatchRecommendRequest(BaseModel):
    unique_ids: list[str] = Field(
        min_length=1,
        max_length=86,
        description="List of market series IDs to retrieve recommendations for (1–86 IDs).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"unique_ids": ["Albany_conventional", "LosAngeles_organic"]}]
        }
    }


class BatchRecommendResponse(BaseModel):
    requested: int = Field(description="Number of unique_ids in the request")
    found: int = Field(description="Number of series for which recommendations were found")
    not_found: list[str] = Field(description="IDs that do not exist in the DataStore")
    results: list[RecommendationResponse]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "requested": 2,
                "found": 1,
                "not_found": ["UNKNOWN_SERIES"],
                "results": ["... RecommendationResponse objects ..."],
            }]
        }
    }


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    unique_id: str
    price: float = Field(gt=0, description="Custom price to test (USD per avocado)")

    model_config = {
        "json_schema_extra": {
            "examples": [{"unique_id": "Albany_conventional", "price": 1.50}]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "unique_id": "Albany_conventional",
                "price": 1.50,
                "predicted_log_volume": 11.92,
                "predicted_volume": 149786.0,
                "current_price": 1.57,
                "current_volume": 86913.0,
            }]
        }
    }
