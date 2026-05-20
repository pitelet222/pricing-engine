"""
uncertainty.py — Conformal prediction intervals and uncertainty-aware pricing.

Extracted from notebooks/05_uncertainty.ipynb.
Extends the point-estimate pricing from pricer.py with probabilistic demand
uncertainty using two complementary approaches:

1. Split-conformal prediction intervals on forecast residuals
   (distribution-free coverage guarantee).
2. Quantile LightGBM (P10 / P50 / P90) for per-price demand distributions,
   enabling three risk-profile pricing strategies.

Constants
---------
QUANTILES             : Demand quantiles trained (P10, P50, P90).
ALPHA                 : Miscoverage rate (0.10 → 90% nominal conformal coverage).
PRICE_MARGIN          : Fractional half-width of the price search range (±30%).
N_GRID                : Grid resolution for revenue curves (300 points).
SEED                  : Random seed for quantile model training.
BASE_QUANTILE_PARAMS  : LightGBM hyperparameters for quantile regression.

Functions
---------
compute_conformal_pi  : Per-series PI stats from CV residuals (split-conformal).
revenue_scenarios     : Revenue arrays at P10/P50/P90 demand over a price grid.
optimize_strategies   : Conservative / balanced / aggressive optimal prices.
compute_risk_metrics  : Downside risk, uplift Sharpe, strategy agreement.
train_quantile_models : Train one LightGBM quantile model per quantile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES: list[float] = [0.10, 0.50, 0.90]
ALPHA: float = 0.10  # 1 - ALPHA = 90% nominal conformal coverage
PRICE_MARGIN: float = 0.30
N_GRID: int = 300
SEED: int = 42

BASE_QUANTILE_PARAMS: dict = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbose": -1,
    "metric": "quantile",
}


def compute_conformal_pi(
    cv_df: pd.DataFrame,
    model_col: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Compute per-series conformal prediction interval statistics from CV residuals.

    Uses split-conformal calibration: residuals r_i = y_i - ŷ_i on the CV
    holdout define empirical quantiles that yield valid prediction intervals
    for new forecasts: [ŷ + q_{α/2}, ŷ + q_{1-α/2}].

    Coverage is nominally 1 - alpha (default 90%).

    Parameters
    ----------
    cv_df     : Cross-validation DataFrame with columns [unique_id, y, model_col].
    model_col : Forecast column name whose residuals are used for calibration.
    alpha     : Miscoverage rate (default 0.10 → 90% nominal coverage).

    Returns
    -------
    DataFrame with columns:
    unique_id, lower_q, upper_q, resid_std, resid_mean, n, pi_width.
    """
    cv_df = cv_df.copy()
    cv_df["residual"] = cv_df["y"] - cv_df[model_col]
    pi_stats = (
        cv_df.groupby("unique_id")["residual"]
        .agg(
            lower_q=lambda r: float(np.quantile(r, alpha / 2)),
            upper_q=lambda r: float(np.quantile(r, 1 - alpha / 2)),
            resid_std=lambda r: float(r.std()),
            resid_mean=lambda r: float(r.mean()),
            n=len,
        )
        .reset_index()
    )
    pi_stats["pi_width"] = pi_stats["upper_q"] - pi_stats["lower_q"]
    return pi_stats


def revenue_scenarios(
    context_row: pd.Series,
    price_grid: np.ndarray,
    quantile_models: dict,
    feature_cols: list[str],
) -> dict[float, np.ndarray]:
    """
    Compute revenue at each price under P10, P50, P90 demand quantiles.

    For each price point the three quantile models yield pessimistic,
    median, and optimistic demand estimates.  Revenue = price × exp(log_demand).

    Parameters
    ----------
    context_row     : Latest feature row for the series being priced.
    price_grid      : 1-D array of candidate prices.
    quantile_models : dict mapping float quantile → trained LightGBM Booster.
    feature_cols    : Ordered list of features expected by each model.

    Returns
    -------
    dict mapping each quantile to a revenue array of the same length as price_grid.
    """
    lag1 = float(context_row["price_lag_1w"])
    rmean4 = float(context_row["price_rmean_4w"])
    rows = []
    for p in price_grid:
        r = context_row.copy()
        r["AveragePrice"] = p
        if lag1 != 0:
            r["price_pct_change_1w"] = (p - lag1) / lag1
        r["price_momentum_4w"] = p - rmean4
        rows.append(r[feature_cols].values)

    X_grid = np.array(rows)
    return {
        q: price_grid * np.exp(model.predict(X_grid))
        for q, model in quantile_models.items()
    }


def optimize_strategies(
    context_row: pd.Series,
    p_mean: float,
    quantile_models: dict,
    feature_cols: list[str],
    margin: float = PRICE_MARGIN,
    n_points: int = N_GRID,
) -> dict:
    """
    Derive conservative, balanced, and aggressive pricing strategies.

    Each strategy optimises revenue under a different demand quantile:
      conservative → P10 (worst plausible demand — guarantees a floor)
      balanced     → P50 (median demand — same as point-estimate pricing)
      aggressive   → P90 (best plausible demand — bets on upside)

    Parameters
    ----------
    context_row     : Latest feature row for the series being priced.
    p_mean          : Historical mean price (used to set grid bounds).
    quantile_models : dict mapping float quantile → trained LightGBM Booster.
    feature_cols    : Ordered list of features expected by each model.
    margin          : Fractional half-width of the price search range (default 30%).
    n_points        : Number of grid points (default 300).

    Returns
    -------
    dict with current_price, rev_p10/p50/p90_current, rev_spread_pct, and for
    each strategy name: opt_price_{name} and rev_uplift_{name}.
    """
    p_min = max(0.50, p_mean * (1 - margin))
    p_max = p_mean * (1 + margin)
    p_curr = float(context_row["AveragePrice"])
    grid = np.linspace(p_min, p_max, n_points)

    revs = revenue_scenarios(context_row, grid, quantile_models, feature_cols)
    curr_revs = revenue_scenarios(context_row, np.array([p_curr]), quantile_models, feature_cols)

    rec: dict = {
        "current_price": round(p_curr, 4),
        "rev_p10_current": round(float(curr_revs[0.10][0]), 2),
        "rev_p50_current": round(float(curr_revs[0.50][0]), 2),
        "rev_p90_current": round(float(curr_revs[0.90][0]), 2),
    }
    p50_curr = max(float(curr_revs[0.50][0]), 1.0)
    rec["rev_spread_pct"] = round(
        (float(curr_revs[0.90][0]) - float(curr_revs[0.10][0])) / p50_curr * 100, 2
    )

    for q, name in [(0.10, "conservative"), (0.50, "balanced"), (0.90, "aggressive")]:
        opt_idx = int(revs[q].argmax())
        opt_p = float(grid[opt_idx])
        opt_r = float(revs[q][opt_idx])
        curr_r = float(curr_revs[q][0])
        pct_rev = (opt_r - curr_r) / curr_r * 100 if curr_r > 0 else 0.0
        rec[f"opt_price_{name}"] = round(opt_p, 4)
        rec[f"rev_uplift_{name}"] = round(pct_rev, 2)

    return rec


def compute_risk_metrics(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a strategy DataFrame with risk metrics.

    Metrics computed
    ----------------
    downside_risk_pct        : (rev_p50 - rev_p10) / rev_p50 at current price.
    uplift_std               : Std of revenue uplift across the three strategies.
    uplift_mean              : Mean of revenue uplift across the three strategies.
    uplift_sharpe            : uplift_mean / uplift_std (uplift per unit of volatility).
    price_direction_*        : Sign of (opt_price - current_price) per strategy.
    strategies_agree         : True when conservative and aggressive agree on direction.

    Parameters
    ----------
    strategy_df : DataFrame with columns rev_p10_current, rev_p50_current,
                  rev_p90_current, opt_price_conservative, opt_price_aggressive,
                  rev_uplift_conservative, rev_uplift_balanced, rev_uplift_aggressive,
                  current_price.

    Returns
    -------
    Copy of strategy_df with the risk columns added.
    """
    df = strategy_df.copy()
    df["downside_risk_pct"] = (
        (df["rev_p50_current"] - df["rev_p10_current"]) / df["rev_p50_current"].clip(lower=1)
    ) * 100

    uplift_cols = ["rev_uplift_conservative", "rev_uplift_balanced", "rev_uplift_aggressive"]
    df["uplift_std"] = df[uplift_cols].std(axis=1)
    df["uplift_mean"] = df[uplift_cols].mean(axis=1)
    df["uplift_sharpe"] = df["uplift_mean"] / df["uplift_std"].clip(lower=0.01)

    df["price_direction_conservative"] = np.sign(
        df["opt_price_conservative"] - df["current_price"]
    )
    df["price_direction_aggressive"] = np.sign(
        df["opt_price_aggressive"] - df["current_price"]
    )
    df["strategies_agree"] = (
        df["price_direction_conservative"] == df["price_direction_aggressive"]
    )
    return df


def train_quantile_models(  # pragma: no cover
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: list[float] = QUANTILES,
    seed: int = SEED,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 40,
) -> dict:
    """
    Train one LightGBM quantile regression model per quantile.

    Each model minimises pinball loss at its assigned quantile, yielding
    P10/P50/P90 demand distributions used by optimize_strategies.

    Parameters
    ----------
    X_train, y_train      : Training features and log_total_volume target.
    X_val,   y_val        : Temporal holdout for early stopping.
    quantiles             : Quantiles to train (default [0.10, 0.50, 0.90]).
    seed                  : Random seed.
    num_boost_round       : Maximum boosting rounds per model.
    early_stopping_rounds : Early stopping patience (rounds without improvement).

    Returns
    -------
    dict mapping each float quantile to its trained lgb.Booster.
    """
    import lightgbm as lgb

    models = {}
    for q in quantiles:
        params = {**BASE_QUANTILE_PARAMS, "alpha": q, "random_state": seed}
        lgb_tr = lgb.Dataset(X_train, label=y_train)
        lgb_va = lgb.Dataset(X_val, label=y_val, reference=lgb_tr)
        models[q] = lgb.train(
            params,
            lgb_tr,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_va],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    return models
