"""
Unit tests for src/models/uncertainty.py.

compute_conformal_pi, compute_risk_metrics, revenue_scenarios, and
optimize_strategies are tested here using lightweight mock models.
train_quantile_models requires lightgbm and is excluded via pragma: no cover.
"""

import numpy as np
import pandas as pd

from src.models.pricer import FEATURE_COLS
from src.models.uncertainty import (
    compute_conformal_pi,
    compute_risk_metrics,
    optimize_strategies,
    revenue_scenarios,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cv_df(n_per_series: int = 20, n_series: int = 3) -> pd.DataFrame:
    """Minimal CV DataFrame with unique_id, y, and a forecast column."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_series):
        uid = f"Series_{i}"
        y = 1.5 + rng.normal(0, 0.1, n_per_series)
        forecast = y + rng.normal(0, 0.05, n_per_series)  # small residuals
        for y_val, f_val in zip(y, forecast):
            rows.append({"unique_id": uid, "y": y_val, "forecast": f_val})
    return pd.DataFrame(rows)


def _make_strategy_df(n: int = 5) -> pd.DataFrame:
    """Minimal strategy DataFrame for compute_risk_metrics."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "unique_id": [f"S{i}" for i in range(n)],
        "current_price": rng.uniform(1.0, 2.5, n),
        "rev_p10_current": rng.uniform(100, 200, n),
        "rev_p50_current": rng.uniform(200, 300, n),
        "rev_p90_current": rng.uniform(300, 400, n),
        "rev_spread_pct": rng.uniform(5, 30, n),
        "opt_price_conservative": rng.uniform(1.0, 2.5, n),
        "opt_price_balanced": rng.uniform(1.0, 2.5, n),
        "opt_price_aggressive": rng.uniform(1.0, 2.5, n),
        "rev_uplift_conservative": rng.uniform(-5, 10, n),
        "rev_uplift_balanced": rng.uniform(-3, 15, n),
        "rev_uplift_aggressive": rng.uniform(0, 20, n),
    })


# ---------------------------------------------------------------------------
# compute_conformal_pi
# ---------------------------------------------------------------------------

def test_compute_conformal_pi_returns_one_row_per_series():
    cv = _make_cv_df(n_series=4)
    result = compute_conformal_pi(cv, model_col="forecast")
    assert len(result) == 4


def test_compute_conformal_pi_output_columns():
    cv = _make_cv_df()
    result = compute_conformal_pi(cv, model_col="forecast")
    expected = {"unique_id", "lower_q", "upper_q", "resid_std", "resid_mean", "n", "pi_width"}
    assert expected.issubset(set(result.columns))


def test_compute_conformal_pi_width_nonnegative():
    cv = _make_cv_df()
    result = compute_conformal_pi(cv, model_col="forecast")
    assert (result["pi_width"] >= 0).all()


def test_compute_conformal_pi_width_equals_upper_minus_lower():
    cv = _make_cv_df()
    result = compute_conformal_pi(cv, model_col="forecast")
    expected_width = result["upper_q"] - result["lower_q"]
    pd.testing.assert_series_equal(
        result["pi_width"].reset_index(drop=True),
        expected_width.reset_index(drop=True),
        check_names=False,
    )


def test_compute_conformal_pi_lower_le_upper():
    cv = _make_cv_df()
    result = compute_conformal_pi(cv, model_col="forecast")
    assert (result["lower_q"] <= result["upper_q"]).all()


def test_compute_conformal_pi_n_matches_series_size():
    n_per = 15
    cv = _make_cv_df(n_per_series=n_per, n_series=2)
    result = compute_conformal_pi(cv, model_col="forecast")
    assert (result["n"] == n_per).all()


def test_compute_conformal_pi_custom_alpha():
    cv = _make_cv_df(n_per_series=100)
    result_90 = compute_conformal_pi(cv, model_col="forecast", alpha=0.10)
    result_80 = compute_conformal_pi(cv, model_col="forecast", alpha=0.20)
    # Narrower alpha=0.20 → narrower interval than alpha=0.10
    assert (result_80["pi_width"] <= result_90["pi_width"] + 1e-9).all()


# ---------------------------------------------------------------------------
# compute_risk_metrics
# ---------------------------------------------------------------------------

def test_compute_risk_metrics_adds_new_columns():
    df = _make_strategy_df()
    result = compute_risk_metrics(df)
    for col in (
        "downside_risk_pct",
        "uplift_std",
        "uplift_mean",
        "uplift_sharpe",
        "price_direction_conservative",
        "price_direction_aggressive",
        "strategies_agree",
    ):
        assert col in result.columns, f"Missing column: {col}"


def test_compute_risk_metrics_does_not_modify_original():
    df = _make_strategy_df()
    df_copy = df.copy()
    compute_risk_metrics(df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_compute_risk_metrics_downside_risk_nonnegative():
    df = _make_strategy_df()
    result = compute_risk_metrics(df)
    assert (result["downside_risk_pct"] >= 0).all()


def test_compute_risk_metrics_strategies_agree_when_same_direction():
    df = pd.DataFrame({
        "unique_id": ["S0"],
        "current_price": [1.50],
        "rev_p10_current": [100.0],
        "rev_p50_current": [200.0],
        "rev_p90_current": [300.0],
        "rev_spread_pct": [10.0],
        "opt_price_conservative": [2.00],   # both above current → agree
        "opt_price_balanced": [2.10],
        "opt_price_aggressive": [2.20],
        "rev_uplift_conservative": [5.0],
        "rev_uplift_balanced": [8.0],
        "rev_uplift_aggressive": [12.0],
    })
    result = compute_risk_metrics(df)
    assert result["strategies_agree"].iloc[0]


def test_compute_risk_metrics_strategies_disagree_when_opposite():
    df = pd.DataFrame({
        "unique_id": ["S0"],
        "current_price": [1.50],
        "rev_p10_current": [100.0],
        "rev_p50_current": [200.0],
        "rev_p90_current": [300.0],
        "rev_spread_pct": [10.0],
        "opt_price_conservative": [1.20],   # conservative → cut
        "opt_price_balanced": [1.60],
        "opt_price_aggressive": [1.80],     # aggressive → raise
        "rev_uplift_conservative": [3.0],
        "rev_uplift_balanced": [6.0],
        "rev_uplift_aggressive": [10.0],
    })
    result = compute_risk_metrics(df)
    assert not result["strategies_agree"].iloc[0]


def test_compute_risk_metrics_price_direction_sign():
    df = pd.DataFrame({
        "unique_id": ["S0", "S1"],
        "current_price": [1.50, 2.00],
        "rev_p10_current": [100.0, 150.0],
        "rev_p50_current": [200.0, 250.0],
        "rev_p90_current": [300.0, 350.0],
        "rev_spread_pct": [10.0, 8.0],
        "opt_price_conservative": [2.00, 1.50],  # +, -
        "opt_price_balanced": [2.10, 1.60],
        "opt_price_aggressive": [2.20, 1.40],
        "rev_uplift_conservative": [5.0, 3.0],
        "rev_uplift_balanced": [8.0, 5.0],
        "rev_uplift_aggressive": [12.0, 7.0],
    })
    result = compute_risk_metrics(df)
    assert result["price_direction_conservative"].iloc[0] > 0   # raise
    assert result["price_direction_conservative"].iloc[1] < 0   # cut


def test_compute_risk_metrics_uplift_sharpe_finite():
    df = _make_strategy_df(n=10)
    result = compute_risk_metrics(df)
    assert result["uplift_sharpe"].notna().all()
    assert np.isfinite(result["uplift_sharpe"]).all()


# ---------------------------------------------------------------------------
# Mock quantile model for revenue_scenarios / optimize_strategies
# ---------------------------------------------------------------------------

class _ConstantQuantileModel:
    """Returns a fixed log-volume regardless of input features."""

    def __init__(self, log_vol: float = 10.5):
        self.log_vol = log_vol

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.full(X.shape[0], self.log_vol)


def _make_quantile_models() -> dict:
    return {
        0.10: _ConstantQuantileModel(log_vol=10.2),
        0.50: _ConstantQuantileModel(log_vol=10.5),
        0.90: _ConstantQuantileModel(log_vol=10.8),
    }


def _make_context_row(price: float = 1.5) -> pd.Series:
    data = {col: 0.0 for col in FEATURE_COLS}
    data["AveragePrice"] = price
    data["price_lag_1w"] = price
    data["price_rmean_4w"] = price
    data["price_pct_change_1w"] = 0.0
    data["price_momentum_4w"] = 0.0
    return pd.Series(data)


# ---------------------------------------------------------------------------
# revenue_scenarios
# ---------------------------------------------------------------------------

def test_revenue_scenarios_returns_all_quantiles():
    qmodels = _make_quantile_models()
    row = _make_context_row()
    grid = np.linspace(0.5, 3.0, 20)
    result = revenue_scenarios(row, grid, qmodels, FEATURE_COLS)
    assert set(result.keys()) == {0.10, 0.50, 0.90}


def test_revenue_scenarios_length_matches_grid():
    qmodels = _make_quantile_models()
    row = _make_context_row()
    grid = np.linspace(0.5, 3.0, 15)
    result = revenue_scenarios(row, grid, qmodels, FEATURE_COLS)
    for q, revs in result.items():
        assert len(revs) == 15, f"quantile {q}: expected 15, got {len(revs)}"


def test_revenue_scenarios_formula():
    log_vol = 10.0
    qmodels = {0.50: _ConstantQuantileModel(log_vol=log_vol)}
    row = _make_context_row()
    grid = np.array([1.0, 2.0, 3.0])
    result = revenue_scenarios(row, grid, qmodels, FEATURE_COLS)
    expected = grid * np.exp(log_vol)
    np.testing.assert_allclose(result[0.50], expected, rtol=1e-6)


def test_revenue_scenarios_nonnegative():
    qmodels = _make_quantile_models()
    row = _make_context_row()
    grid = np.linspace(0.5, 3.0, 10)
    result = revenue_scenarios(row, grid, qmodels, FEATURE_COLS)
    for q, revs in result.items():
        assert (revs >= 0).all(), f"Negative revenue for quantile {q}"


# ---------------------------------------------------------------------------
# optimize_strategies
# ---------------------------------------------------------------------------

def test_optimize_strategies_returns_required_keys():
    qmodels = _make_quantile_models()
    row = _make_context_row(price=1.5)
    result = optimize_strategies(row, p_mean=1.5, quantile_models=qmodels, feature_cols=FEATURE_COLS)
    required = {
        "current_price",
        "rev_p10_current", "rev_p50_current", "rev_p90_current",
        "rev_spread_pct",
        "opt_price_conservative", "opt_price_balanced", "opt_price_aggressive",
        "rev_uplift_conservative", "rev_uplift_balanced", "rev_uplift_aggressive",
    }
    assert required.issubset(result.keys())


def test_optimize_strategies_current_price_matches_row():
    price = 1.75
    qmodels = _make_quantile_models()
    row = _make_context_row(price=price)
    result = optimize_strategies(row, p_mean=price, quantile_models=qmodels, feature_cols=FEATURE_COLS)
    assert result["current_price"] == round(price, 4)


def test_optimize_strategies_optimal_prices_within_bounds():
    p_mean = 1.5
    margin = 0.30
    qmodels = _make_quantile_models()
    row = _make_context_row(price=p_mean)
    result = optimize_strategies(
        row, p_mean=p_mean, quantile_models=qmodels,
        feature_cols=FEATURE_COLS, margin=margin,
    )
    p_min = max(0.50, p_mean * (1 - margin))
    p_max = p_mean * (1 + margin)
    for strategy in ("conservative", "balanced", "aggressive"):
        opt_p = result[f"opt_price_{strategy}"]
        assert p_min <= opt_p <= p_max + 1e-9, f"{strategy}: {opt_p} not in [{p_min}, {p_max}]"


def test_optimize_strategies_rev_spread_pct_nonneg():
    qmodels = _make_quantile_models()
    row = _make_context_row()
    result = optimize_strategies(row, p_mean=1.5, quantile_models=qmodels, feature_cols=FEATURE_COLS)
    assert result["rev_spread_pct"] >= 0
