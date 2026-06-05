"""
forecaster.py — Statistical and neural price forecasting wrappers.

Extracted from notebooks/03_forecasting.ipynb.
Provides helpers for converting the feature matrix to Nixtla long format,
generating future exogenous features, and computing inverse-MAE ensemble weights.

The heavier StatsForecast and NeuralForecast wrappers are deferred to
run_statistical_forecast / run_neural_forecast and import those packages lazily
so the module can be imported in environments where only numpy/pandas are present.

Constants
---------
H         : Forecast horizon (12 weeks ≈ 3 months).
SEASON    : Annual seasonality period for weekly data (52 weeks).
SEED      : Global random seed for reproducibility.
FUTR_EXOG : Future-known exogenous feature names (calendar features).
STAT_EXOG : Static per-series exogenous feature names.

Functions
---------
build_nixtla_df         : Convert avocado feature DataFrame to Nixtla long format.
build_future_exog       : Generate future exogenous rows for H steps per series.
compute_ensemble_weights: Inverse-MAE weights from cross-validation predictions.
run_statistical_forecast: Fit MSTL-wrapped statistical models with StatsForecast.
run_neural_forecast     : Fit NHITS with NeuralForecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H: int = 12
SEASON: int = 52
SEED: int = 42

# Calendar features known at prediction time — used as future exogenous inputs.
FUTR_EXOG: list[str] = [
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "is_holiday_week",
]

# Constant per-series structural features — passed as static exogenous inputs.
STAT_EXOG: list[str] = [
    "is_organic",
    "region_price_level",
    "region_volume_level",
]


def build_nixtla_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the avocado feature DataFrame to Nixtla long format.

    Creates unique_id = "{region}_{organic|conventional}", renames Date → ds
    and AveragePrice → y, and selects only the columns required by StatsForecast
    and NeuralForecast.

    Parameters
    ----------
    df : DataFrame from features.build_features (avocado_features.csv schema).

    Returns
    -------
    DataFrame sorted by (unique_id, ds) with columns:
    unique_id, ds, y, month_sin, month_cos, week_sin, week_cos,
    is_holiday_week, is_organic, region_price_level, region_volume_level.
    """
    df = df.copy()
    df["unique_id"] = df["region"] + "_" + df["is_organic"].map({0: "conventional", 1: "organic"})
    nf_df = df.rename(columns={"Date": "ds", "AveragePrice": "y"})
    cols = ["unique_id", "ds", "y"] + FUTR_EXOG + STAT_EXOG
    return nf_df[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_future_exog(nf_df: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """
    Generate future exogenous feature rows for the next h weeks per series.

    Computes deterministic calendar features (FUTR_EXOG) for each series's
    h future dates.  is_holiday_week is set to 0 for all future dates — a
    production system would compute actual future holiday flags here.

    Parameters
    ----------
    nf_df : Nixtla-format DataFrame (output of build_nixtla_df).
    h     : Forecast horizon in weeks (default H = 12).

    Returns
    -------
    DataFrame with columns: unique_id, ds, month_sin, month_cos,
    week_sin, week_cos, is_holiday_week.
    """
    last_dates = nf_df.groupby("unique_id")["ds"].max()
    rows = []
    for uid, last_date in last_dates.items():
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=h, freq="W")
        for d in future_dates:
            iso_week = d.isocalendar()[1]
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "month_sin": np.sin(2 * np.pi * d.month / 12),
                    "month_cos": np.cos(2 * np.pi * d.month / 12),
                    "week_sin": np.sin(2 * np.pi * iso_week / 52),
                    "week_cos": np.cos(2 * np.pi * iso_week / 52),
                    "is_holiday_week": 0,
                }
            )
    return pd.DataFrame(rows)


def compute_ensemble_weights(all_cv: pd.DataFrame, all_models: list[str]) -> dict[str, float]:
    """
    Compute inverse-MAE ensemble weights from cross-validation predictions.

    Each model's weight is proportional to 1 / CV_MAE, normalised to sum to 1.
    Models with lower CV error receive higher weight in the ensemble.

    Parameters
    ----------
    all_cv     : Merged CV DataFrame with columns [unique_id, ds, y, *model_cols].
    all_models : Ordered list of model column names in all_cv.

    Returns
    -------
    dict mapping each model name to its normalised weight (all weights sum to 1).
    """
    inv_mae = {m: 1.0 / float(np.abs(all_cv["y"] - all_cv[m]).mean()) for m in all_models}
    total = sum(inv_mae.values())
    return {m: v / total for m, v in inv_mae.items()}


def run_statistical_forecast(  # pragma: no cover
    sf_df: pd.DataFrame,
    h: int = H,
    season: int = SEASON,
) -> pd.DataFrame:
    """
    Fit MSTL-wrapped statistical models and return cross-validation predictions.

    Models trained: SeasonalNaive, MSTL_ARIMA, MSTL_ETS, MSTL_Theta.
    Uses StatsForecast cross_validation(n_windows=1) — last h weeks as holdout.

    MSTL wrapping is required because fitting AutoARIMA/AutoETS/AutoTheta with
    season_length=52 directly on ~3 years of weekly data causes flat forecasts.
    STL separates the 52-week seasonal pattern robustly; the inner forecaster
    then models only the smoother de-seasonalised residual.

    Parameters
    ----------
    sf_df  : Nixtla univariate DataFrame (unique_id, ds, y).
    h      : Forecast horizon in weeks.
    season : Seasonal period (52 for weekly data).

    Returns
    -------
    DataFrame with columns: unique_id, ds, cutoff, y,
    SeasonalNaive, MSTL_ARIMA, MSTL_ETS, MSTL_Theta.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

    models = [
        SeasonalNaive(season_length=season),
        MSTL(season_length=season, trend_forecaster=AutoARIMA(season_length=1), alias="MSTL_ARIMA"),
        MSTL(season_length=season, trend_forecaster=AutoETS(model="ZZN"), alias="MSTL_ETS"),
        MSTL(season_length=season, trend_forecaster=AutoTheta(season_length=1), alias="MSTL_Theta"),
    ]
    sf = StatsForecast(models=models, freq="W", n_jobs=-1)
    cv = sf.cross_validation(df=sf_df, h=h, n_windows=1)
    return cv.reset_index()


def run_neural_forecast(  # pragma: no cover
    nf_df: pd.DataFrame,
    h: int = H,
    season: int = SEASON,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Fit NHITS and return cross-validation predictions.

    NHITS (Neural Hierarchical Interpolation for Time Series) is trained as a
    global model across all series simultaneously.  Uses future-known calendar
    exogenous features (FUTR_EXOG) and static series-level features (STAT_EXOG).

    Parameters
    ----------
    nf_df  : Nixtla DataFrame from build_nixtla_df (includes exog columns).
    h      : Forecast horizon in weeks.
    season : Used implicitly via input_size = 2 * h.
    seed   : Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns: unique_id, ds, cutoff, NHITS, y.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    nhits = NHITS(
        h=h,
        input_size=2 * h,
        futr_exog_list=FUTR_EXOG,
        stat_exog_list=STAT_EXOG,
        scaler_type="robust",
        max_steps=500,
        learning_rate=1e-3,
        early_stop_patience_steps=10,
        val_check_steps=50,
        random_seed=seed,
        accelerator="auto",
        enable_progress_bar=True,
    )
    static_df = nf_df[["unique_id"] + STAT_EXOG].drop_duplicates("unique_id")
    nf = NeuralForecast(models=[nhits], freq="W")
    cv = nf.cross_validation(
        df=nf_df.drop(columns=STAT_EXOG),
        static_df=static_df,
        n_windows=1,
        val_size=h,
    )
    return cv.reset_index()
