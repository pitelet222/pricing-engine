"""
pricer.py — LightGBM demand model training and revenue-optimal pricing.

Extracted from notebooks/04_pricing.ipynb.
Provides the demand modelling and pricing optimisation layer: train a LightGBM
Booster to predict log(demand), then find the price that maximises expected
weekly revenue via dense grid search.

Constants
---------
FEATURE_COLS  : Ordered list of 26 features expected by the demand model.
TARGET_COL    : 'log_total_volume' — log1p(Total Volume) as regression target.
PRICE_MARGIN  : Fractional half-width of the price search range (±30%).
N_GRID        : Number of candidate prices in the grid search (300).
SEED          : Random seed for LightGBM reproducibility.
LGB_PARAMS    : Default LightGBM training hyperparameters.

Functions
---------
compute_elasticity : Point price elasticity via central difference.
revenue_curve      : Expected revenue at each price in a grid.
optimize_price     : Revenue-maximising price within ±PRICE_MARGIN bounds.
train_demand_model : Train LightGBM with early stopping on a temporal holdout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    # Price — the lever we optimise
    "AveragePrice",
    # Structural
    "is_organic",
    "region_encoded",
    "region_price_level",
    "region_volume_level",
    # Calendar
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "is_holiday_week",
    "month",
    "quarter",
    # Price history
    "price_lag_1w",
    "price_lag_2w",
    "price_lag_4w",
    "price_rmean_4w",
    "price_rmean_8w",
    "price_rmean_12w",
    "price_rstd_4w",
    "price_pct_change_1w",
    "price_momentum_4w",
    # Product mix ratios — raw volumes deliberately excluded to avoid
    # leaking the target (log_total_volume ≈ sum of log PLU volumes)
    "plu_share_4046",
    "plu_share_4225",
    "plu_share_4770",
    "bag_share",
    "bag_size_ratio",
]
TARGET_COL: str = "log_total_volume"
PRICE_MARGIN: float = 0.30
N_GRID: int = 300
SEED: int = 42

LGB_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
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
}


def compute_elasticity(
    row: pd.Series,
    model: Any,
    feature_cols: list[str] = FEATURE_COLS,
    delta: float = 0.01,
) -> float:
    """
    Numerically estimate point price elasticity at the row's current price.

    Uses central difference: ε ≈ [(Q₊ - Q₋) / (2·ΔP)] × (P / Q).
    price_pct_change_1w and price_momentum_4w are recalculated at each
    perturbation so that price-derived features stay internally consistent.

    Parameters
    ----------
    row         : Feature row with all feature_cols present.
    model       : Trained LightGBM Booster (or any object with .predict(X)).
    feature_cols: Ordered list of features expected by model.
    delta       : Fractional price perturbation (default 1%).

    Returns
    -------
    Price elasticity ε (negative for normal goods; < -1 = elastic demand).
    """
    p = float(row["AveragePrice"])
    dp = p * delta

    def _predict_volume(price: float) -> float:
        r = row.copy()
        r["AveragePrice"] = price
        lag1 = float(r["price_lag_1w"])
        if lag1 != 0:
            r["price_pct_change_1w"] = (price - lag1) / lag1
        r["price_momentum_4w"] = price - float(r["price_rmean_4w"])
        return float(np.exp(model.predict(r[feature_cols].values.reshape(1, -1))[0]))

    q_up = _predict_volume(p + dp)
    q_dn = _predict_volume(p - dp)
    q = _predict_volume(p)
    dq_dp = (q_up - q_dn) / (2 * dp)
    return float(dq_dp * (p / q))


def revenue_curve(
    context_row: pd.Series,
    price_grid: np.ndarray,
    model: Any,
    feature_cols: list[str] = FEATURE_COLS,
) -> np.ndarray:
    """
    Compute expected weekly revenue at each price in price_grid.

    All features are held fixed at their current-context values except
    AveragePrice, price_pct_change_1w, and price_momentum_4w — which are
    recalculated at each price point to maintain feature consistency.

    Revenue = price × exp(predicted log_total_volume).

    Parameters
    ----------
    context_row : Latest feature row for the series being priced.
    price_grid  : 1-D array of candidate prices.
    model       : Trained LightGBM Booster.
    feature_cols: Ordered list of features expected by model.

    Returns
    -------
    np.ndarray of revenue values, same length as price_grid.
    """
    lag1 = float(context_row["price_lag_1w"])
    rmean4 = float(context_row["price_rmean_4w"])
    revenues = []
    for p in price_grid:
        r = context_row.copy()
        r["AveragePrice"] = p
        if lag1 != 0:
            r["price_pct_change_1w"] = (p - lag1) / lag1
        r["price_momentum_4w"] = p - rmean4
        log_vol = float(model.predict(r[feature_cols].values.reshape(1, -1))[0])
        revenues.append(p * np.exp(log_vol))
    return np.array(revenues)


def optimize_price(
    context_row: pd.Series,
    p_mean: float,
    model: Any,
    feature_cols: list[str] = FEATURE_COLS,
    margin: float = PRICE_MARGIN,
    n_points: int = N_GRID,
) -> dict:
    """
    Find the revenue-maximising price within [p_mean*(1-margin), p_mean*(1+margin)].

    Uses a dense grid search over n_points candidate prices — transparent,
    auditable, and fast enough for 86 series at 300 points each.

    Parameters
    ----------
    context_row : Latest feature row for the series being priced.
    p_mean      : Historical mean price for the series (sets grid bounds).
    model       : Trained LightGBM Booster.
    feature_cols: Ordered list of features expected by model.
    margin      : Fractional half-width of the price search range (default 30%).
    n_points    : Number of grid points (default 300).

    Returns
    -------
    dict with keys:
      current_price   : Current price at context_row["AveragePrice"].
      optimal_price   : Revenue-maximising price within bounds.
      current_revenue : Expected revenue at current price.
      optimal_revenue : Expected revenue at optimal price.
      price_grid      : np.ndarray of all candidate prices.
      revenue_curve   : np.ndarray of revenue values aligned to price_grid.
    """
    p_min = max(0.50, p_mean * (1 - margin))
    p_max = p_mean * (1 + margin)
    p_curr = float(context_row["AveragePrice"])
    grid = np.linspace(p_min, p_max, n_points)
    revs = revenue_curve(context_row, grid, model, feature_cols)
    curr_rev = float(revenue_curve(context_row, np.array([p_curr]), model, feature_cols)[0])
    best_idx = int(revs.argmax())
    return {
        "current_price": round(p_curr, 4),
        "optimal_price": round(float(grid[best_idx]), 4),
        "current_revenue": round(curr_rev, 2),
        "optimal_revenue": round(float(revs[best_idx]), 2),
        "price_grid": grid,
        "revenue_curve": revs,
    }


def train_demand_model(  # pragma: no cover
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = SEED,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> Any:
    """
    Train a LightGBM demand model with early stopping on a temporal holdout.

    Uses the LGB_PARAMS hyperparameters validated in notebook 04 (best round ≈ 245).
    Early stopping prevents overfitting while keeping the API serving model
    as accurate as possible.

    Parameters
    ----------
    X_train, y_train          : Training features and log_total_volume target.
    X_val,   y_val            : Temporal holdout for early stopping.
    seed                      : Random seed for reproducibility.
    num_boost_round           : Maximum boosting rounds.
    early_stopping_rounds     : Stop if val RMSE doesn't improve for this many rounds.

    Returns
    -------
    Trained lgb.Booster at its best iteration.
    """
    import lightgbm as lgb

    params = {**LGB_PARAMS, "random_state": seed}
    lgb_tr = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns))
    lgb_va = lgb.Dataset(X_val, label=y_val, feature_name=list(X_val.columns), reference=lgb_tr)
    return lgb.train(
        params,
        lgb_tr,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
