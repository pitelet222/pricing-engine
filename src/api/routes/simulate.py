"""POST /v1/simulate — live LightGBM inference for a custom price."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends

from src.api.rate_limit import check_rate_limit
from src.api.routes._deps import DataStoreDep, _require_series
from src.api.schemas import SimulateRequest, SimulateResponse

router = APIRouter()


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    dependencies=[Depends(check_rate_limit)],
)
def simulate(req: SimulateRequest, ds: DataStoreDep) -> SimulateResponse:
    """
    Run live LightGBM inference for a custom price on one series.

    The most recent feature row for the series is used as context; only
    AveragePrice is swapped out. All other features (lags, rolling stats,
    calendar, region encodings) remain at their last observed values.

    This answers: "If I set price to X today, what demand would the model predict?"
    """
    _require_series(ds, req.unique_id)

    ctx_row = ds.latest_ctx_df[ds.latest_ctx_df["unique_id"] == req.unique_id].iloc[0].copy()

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
