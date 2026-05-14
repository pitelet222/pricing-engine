"""
loader.py — DataStore: loads all pre-computed artefacts once at API startup.

All heavy computation happened in the notebooks. This module's only job is to
read those results from disk and hold them in a single, typed container so the
API layer never touches the filesystem at request time.

The one exception is /simulate, which runs live LightGBM inference. For that,
we pre-compute `latest_ctx_df` here: one feature row per series at the most
recent available date, ready to have AveragePrice swapped out per request.
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Filesystem roots — resolved relative to this file so the API works from any
# working directory.
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parents[2]   # pricing-engine/
_OUTPUTS = _ROOT / "data" / "outputs"
_PROCESSED = _ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------
@dataclass
class DataStore:
    """
    Single container for all artefacts the API serves.

    Every attribute is loaded once inside FastAPI's lifespan context manager
    and held in memory for the lifetime of the process.

    Attributes
    ----------
    forecast_df : DataFrame (1 032 × ~12)
        12-week point forecasts per series. Models: Ensemble_weighted, MSTL_ETS,
        MSTL_ARIMA, MSTL_Theta, NHITS, NBEATSx, DLinear, SeasonalNaive.
        Indexed by unique_id + ds.
    pi_df : DataFrame (86 × 7)
        Per-series conformal prediction-interval stats (lower_q, upper_q,
        resid_std, pi_width …). Applied to Ensemble_weighted forecasts.
    recommendations_df : DataFrame (86 × 9)
        Optimal price, revenue uplift, price elasticity per series.
    uncertainty_df : DataFrame (86 × 20)
        Conservative / balanced / aggressive strategy outputs plus risk metrics
        (rev_spread_pct, downside_risk_pct, uplift_sharpe, strategies_agree …).
    shap_drivers_df : DataFrame (258 × 7)
        Top-3 SHAP drivers per series (86 × 3 rows). Columns: unique_id,
        driver_rank, feature, shap_value, abs_shap_value, direction,
        feature_value.
    latest_ctx_df : DataFrame (86 × 44)
        Most-recent feature row per series from avocado_features.csv, with
        unique_id and region_encoded added. Used by /simulate for live
        inference: caller patches AveragePrice, everything else is fixed.
    model_bundle : dict
        Keys — 'model': trained LightGBM Booster, 'features': ordered list of
        26 feature names, 'target': 'log_total_volume'.
    series_meta : dict[str, dict]
        Keyed by unique_id → {'region': str, 'avocado_type': str}.
        Derived at load time so routes never have to parse unique_id strings.
    """

    forecast_df: pd.DataFrame
    pi_df: pd.DataFrame
    recommendations_df: pd.DataFrame
    uncertainty_df: pd.DataFrame
    shap_drivers_df: pd.DataFrame
    latest_ctx_df: pd.DataFrame
    model_bundle: dict
    series_meta: dict[str, dict]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require(path: pathlib.Path) -> pathlib.Path:
    """Raise a clear FileNotFoundError if a required artefact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required artefact not found: {path}\n"
            "Run the notebooks in order (01 → 06) to generate all outputs."
        )
    return path


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_require(_OUTPUTS / name))


def _build_latest_ctx(
    features_path: pathlib.Path,
    label_encoder,
) -> pd.DataFrame:
    """
    Return one feature row per series at the most recent available date.

    Steps
    -----
    1. Load avocado_features.csv (14 190 rows, 42 columns).
    2. Construct unique_id as "{region}_{organic|conventional}" — matching the
       format used in every output artefact.
    3. Add region_encoded by re-applying the same LabelEncoder saved at
       training time. Using a different encoder would corrupt the model inputs.
    4. Keep only the last row per series (sorted by Date).
    """
    df = pd.read_csv(features_path, parse_dates=["Date"])

    df["unique_id"] = (
        df["region"] + "_"
        + df["is_organic"].map({1: "organic", 0: "conventional"})
    )

    # Must use the training-time encoder so integer codes match what the model saw.
    df["region_encoded"] = label_encoder.transform(df["region"])

    latest = (
        df.sort_values("Date")
        .groupby("unique_id", group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def _build_series_meta(recommendations_df: pd.DataFrame) -> dict[str, dict]:
    """
    Build a lookup dict so routes never need to parse unique_id strings.

    unique_id format: "{region}_{organic|conventional}"
    is_organic column in recommendations_df: 1 or 0.
    """
    meta: dict[str, dict] = {}
    for _, row in recommendations_df.iterrows():
        uid: str = row["unique_id"]
        region = uid.rsplit("_", 1)[0]
        avocado_type = "organic" if int(row["is_organic"]) == 1 else "conventional"
        meta[uid] = {"region": region, "avocado_type": avocado_type}
    return meta


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_datastore() -> DataStore:
    """
    Load every artefact from disk and return a fully populated DataStore.

    This function is called exactly once, inside FastAPI's lifespan context
    manager. After that, the DataStore is stored in app.state and injected
    into routes via a dependency function.

    Raises
    ------
    FileNotFoundError
        If any required file is missing. The message names the missing path
        and tells the user which notebook generates it.
    """
    label_encoder = joblib.load(_require(_OUTPUTS / "region_label_encoder.pkl"))

    # demand_model.pkl was saved with plain pickle (not joblib) in notebook 02.
    with open(_require(_OUTPUTS / "demand_model.pkl"), "rb") as fh:
        model_bundle = pickle.load(fh)

    latest_ctx_df = _build_latest_ctx(
        _require(_PROCESSED / "avocado_features.csv"),
        label_encoder,
    )

    recommendations_df = _load_csv("pricing_recommendations.csv")

    return DataStore(
        forecast_df=_load_csv("forecast_future.csv"),
        pi_df=_load_csv("conformal_pi_stats.csv"),
        recommendations_df=recommendations_df,
        uncertainty_df=_load_csv("uncertainty_pricing.csv"),
        shap_drivers_df=_load_csv("shap_top_drivers.csv"),
        latest_ctx_df=latest_ctx_df,
        model_bundle=model_bundle,
        series_meta=_build_series_meta(recommendations_df),
    )
