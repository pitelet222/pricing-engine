"""
routes.py — All API endpoint handlers.

Dependency pattern: get_datastore() reads from request.app.state, which is
set by the lifespan in main.py. Using Request here avoids a circular import
between routes.py and main.py.

Response cache
--------------
The four read-only endpoints (/series, /forecast, /recommend, /explain) serve
data that is static between process restarts — it's all pre-computed in the
notebooks. A module-level dict caches the first computed result per unique_id
so subsequent calls skip all DataFrame work and return the stored object directly.
The cache is never invalidated during a process's lifetime; to refresh, restart
the server (which re-runs the lifespan and reloads the DataStore).

Cache hit/miss events are recorded to Prometheus counters so operators can
track cache efficiency over time (see src/api/metrics.py).
"""
from __future__ import annotations

import math
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.metrics import CACHE_HITS, CACHE_MISSES
from src.api.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    ExplainResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    RecommendationResponse,
    SeriesItem,
    SeriesListResponse,
    ShapDriver,
    SimulateRequest,
    SimulateResponse,
)
from src.config import settings
from src.data.loader import DataStore

router = APIRouter()

# Module-level response cache — keyed by "<endpoint>:<unique_id>" or "<endpoint>".
_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Shared dependency
# ---------------------------------------------------------------------------

def get_datastore(request: Request) -> DataStore:
    return request.app.state.datastore


DataStoreDep = Annotated[DataStore, Depends(get_datastore)]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_hit(endpoint: str) -> None:
    CACHE_HITS.labels(endpoint=endpoint).inc()


def _cache_miss(endpoint: str) -> None:
    CACHE_MISSES.labels(endpoint=endpoint).inc()


# ---------------------------------------------------------------------------
# Series helper
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
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(ds: DataStoreDep) -> HealthResponse:
    """
    Liveness / readiness check.

    Returns HTTP 200 with a JSON body as long as the DataStore loaded
    successfully. Suitable for Docker and Kubernetes health probes.
    """
    return HealthResponse(
        status="ok",
        series_loaded=len(ds.series_meta),
        version=settings.api_version,
    )


# ---------------------------------------------------------------------------
# GET /series
# ---------------------------------------------------------------------------

@router.get("/series", response_model=SeriesListResponse)
def list_series(ds: DataStoreDep) -> SeriesListResponse:
    """List all 86 avocado market series with region and type metadata."""
    _ep = "/series"
    if _ep in _CACHE:
        _cache_hit(_ep)
        return _CACHE[_ep]

    _cache_miss(_ep)
    items = [
        SeriesItem(
            unique_id=uid,
            region=meta["region"],
            avocado_type=meta["avocado_type"],
        )
        for uid, meta in ds.series_meta.items()
    ]
    result = SeriesListResponse(total=len(items), series=items)
    _CACHE[_ep] = result
    return result


# ---------------------------------------------------------------------------
# GET /forecast/{unique_id}
# ---------------------------------------------------------------------------

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
        return _CACHE[cache_key]

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


# ---------------------------------------------------------------------------
# GET /recommend/{unique_id}
# ---------------------------------------------------------------------------

@router.get("/recommend/{unique_id}", response_model=RecommendationResponse)
def get_recommendation(unique_id: str, ds: DataStoreDep) -> RecommendationResponse:
    """
    Return the revenue-maximising price recommendation for one series.

    Optimal price and revenue uplift are computed via grid search (±30% bounds)
    using the LightGBM demand model trained in notebook 04.
    """
    _require_series(ds, unique_id)
    return _build_recommendation(ds, unique_id)


def _build_recommendation(ds: DataStore, unique_id: str) -> RecommendationResponse:
    """Build (or return cached) RecommendationResponse for one series."""
    _ep = "/recommend/{unique_id}"
    cache_key = f"recommend:{unique_id}"

    if cache_key in _CACHE:
        _cache_hit(_ep)
        return _CACHE[cache_key]

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


# ---------------------------------------------------------------------------
# POST /batch-recommend
# ---------------------------------------------------------------------------

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
    _ep = "/explain/{unique_id}"
    cache_key = f"explain:{unique_id}"

    if cache_key in _CACHE:
        _cache_hit(_ep)
        return _CACHE[cache_key]

    _cache_miss(_ep)
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
    result = ExplainResponse(unique_id=unique_id, drivers=drivers)
    _CACHE[cache_key] = result
    return result


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
