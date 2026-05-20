"""
Unit tests for src/models/pricer.py.

Uses a _LogLinearMockModel that returns (intercept - coeff * price) as
log-volume so that revenue, elasticity, and optimization can be validated
without requiring a trained LightGBM model.

For a log-linear demand model log(Q) = a - b*P:
  - Revenue R = P * exp(a - b*P) peaks at P* = 1/b (when dR/dP = 0).
  - Price elasticity ε ≈ -b * P (central difference approximation).
"""

import numpy as np
import pandas as pd

from src.models.pricer import FEATURE_COLS, compute_elasticity, optimize_price, revenue_curve

# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------

class _LogLinearMockModel:
    """
    Returns intercept - coeff * AveragePrice as log-volume prediction.

    AveragePrice is at FEATURE_COLS[0].  Revenue = P * exp(a - b*P) peaks
    at P* = 1/b.  Elasticity = -b * P (exact for central difference as dp→0).
    """

    def __init__(self, intercept: float = 11.0, coeff: float = 0.5):
        self.intercept = intercept
        self.coeff = coeff

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        prices = X[:, 0] if X.ndim == 2 else X[np.newaxis, 0]
        return self.intercept - self.coeff * prices


def _make_context_row(price: float = 1.5) -> pd.Series:
    """Build a minimal feature row with all FEATURE_COLS present."""
    data = {col: 0.0 for col in FEATURE_COLS}
    data["AveragePrice"] = price
    data["price_lag_1w"] = price
    data["price_rmean_4w"] = price
    data["price_pct_change_1w"] = 0.0
    data["price_momentum_4w"] = 0.0
    return pd.Series(data)


# ---------------------------------------------------------------------------
# revenue_curve
# ---------------------------------------------------------------------------

def test_revenue_curve_same_length_as_grid():
    model = _LogLinearMockModel()
    row = _make_context_row()
    grid = np.linspace(0.5, 3.0, 100)
    revs = revenue_curve(row, grid, model, FEATURE_COLS)
    assert len(revs) == 100


def test_revenue_curve_formula():
    model = _LogLinearMockModel(intercept=10.0, coeff=0.0)  # constant log-vol
    row = _make_context_row(price=2.0)
    grid = np.array([1.0, 2.0, 3.0])
    revs = revenue_curve(row, grid, model, FEATURE_COLS)
    # revenue = price * exp(10) when coeff=0
    expected = grid * np.exp(10.0)
    np.testing.assert_allclose(revs, expected, rtol=1e-6)


def test_revenue_curve_returns_ndarray():
    model = _LogLinearMockModel()
    row = _make_context_row()
    grid = np.linspace(0.5, 2.5, 50)
    result = revenue_curve(row, grid, model, FEATURE_COLS)
    assert isinstance(result, np.ndarray)


def test_revenue_curve_nonnegative():
    model = _LogLinearMockModel()
    row = _make_context_row()
    grid = np.linspace(0.5, 3.0, 30)
    revs = revenue_curve(row, grid, model, FEATURE_COLS)
    assert (revs >= 0).all()


# ---------------------------------------------------------------------------
# optimize_price
# ---------------------------------------------------------------------------

def test_optimize_price_returns_required_keys():
    model = _LogLinearMockModel()
    row = _make_context_row(price=1.5)
    result = optimize_price(row, p_mean=1.5, model=model, feature_cols=FEATURE_COLS)
    required = {
        "current_price", "optimal_price", "current_revenue",
        "optimal_revenue", "price_grid", "revenue_curve",
    }
    assert required.issubset(result.keys())


def test_optimize_price_within_bounds():
    model = _LogLinearMockModel()
    p_mean = 1.5
    margin = 0.30
    row = _make_context_row(price=p_mean)
    result = optimize_price(row, p_mean=p_mean, model=model, feature_cols=FEATURE_COLS)
    p_min = max(0.50, p_mean * (1 - margin))
    p_max = p_mean * (1 + margin)
    assert p_min <= result["optimal_price"] <= p_max + 1e-9


def test_optimize_price_optimal_ge_current_revenue():
    model = _LogLinearMockModel()
    row = _make_context_row(price=2.5)
    result = optimize_price(row, p_mean=2.0, model=model, feature_cols=FEATURE_COLS)
    assert result["optimal_revenue"] >= result["current_revenue"] - 1e-9


def test_optimize_price_grid_length():
    model = _LogLinearMockModel()
    row = _make_context_row(price=1.5)
    result = optimize_price(
        row, p_mean=1.5, model=model, feature_cols=FEATURE_COLS, n_points=50
    )
    assert len(result["price_grid"]) == 50
    assert len(result["revenue_curve"]) == 50


def test_optimize_price_prices_are_rounded():
    model = _LogLinearMockModel()
    row = _make_context_row(price=1.5)
    result = optimize_price(row, p_mean=1.5, model=model, feature_cols=FEATURE_COLS)
    # current_price and optimal_price should be rounded to 4 decimals
    assert round(result["current_price"], 4) == result["current_price"]
    assert round(result["optimal_price"], 4) == result["optimal_price"]


# ---------------------------------------------------------------------------
# compute_elasticity
# ---------------------------------------------------------------------------

def test_compute_elasticity_negative_for_downward_demand():
    model = _LogLinearMockModel(coeff=0.5)
    row = _make_context_row(price=1.5)
    elasticity = compute_elasticity(row, model, FEATURE_COLS)
    assert elasticity < 0, "Demand decreases with price; elasticity must be negative"


def test_compute_elasticity_approximate_value():
    # For log-linear model: ε ≈ -coeff * price
    coeff = 0.5
    price = 2.0
    model = _LogLinearMockModel(coeff=coeff)
    row = _make_context_row(price=price)
    elasticity = compute_elasticity(row, model, FEATURE_COLS, delta=0.001)
    expected = -coeff * price
    assert abs(elasticity - expected) < 0.05  # central diff approximation


def test_compute_elasticity_returns_float():
    model = _LogLinearMockModel()
    row = _make_context_row(price=1.5)
    result = compute_elasticity(row, model, FEATURE_COLS)
    assert isinstance(result, float)
