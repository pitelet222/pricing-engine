"""
Unit tests for src/evaluation/metrics.py.

All tests are pure numerical — no fixtures or external dependencies.
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Perfect forecast
# ---------------------------------------------------------------------------


def test_mae_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_metrics(y, y)
    assert result["MAE"] == pytest.approx(0.0)


def test_rmse_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_metrics(y, y)
    assert result["RMSE"] == pytest.approx(0.0)


def test_mape_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_metrics(y, y)
    assert result["MAPE"] == pytest.approx(0.0)


def test_smape_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_metrics(y, y)
    assert result["SMAPE"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------


def test_mae_known_values():
    y_true = np.array([3.0, 5.0])
    y_pred = np.array([2.0, 6.0])
    result = compute_metrics(y_true, y_pred)
    assert result["MAE"] == pytest.approx(1.0)


def test_rmse_known_values():
    y_true = np.array([3.0, 5.0])
    y_pred = np.array([2.0, 6.0])
    result = compute_metrics(y_true, y_pred)
    # errors: [-1, +1], MSE = 1.0, RMSE = 1.0
    assert result["RMSE"] == pytest.approx(1.0)


def test_mape_returns_percentage_not_fraction():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    result = compute_metrics(y_true, y_pred)
    # 10% error — MAPE should be 10.0, not 0.10
    assert result["MAPE"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_rmse_ge_mae():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(1, 10, 50)
    y_pred = rng.uniform(1, 10, 50)
    result = compute_metrics(y_true, y_pred)
    assert result["RMSE"] >= result["MAE"] - 1e-12


def test_smape_bounded_by_200():
    y_true = np.array([1.0, 0.0, 5.0])
    y_pred = np.array([0.0, 1.0, 100.0])
    result = compute_metrics(y_true, y_pred)
    assert result["SMAPE"] <= 200.0 + 1e-9


def test_smape_symmetric():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([4.0, 2.0, 3.0])
    r1 = compute_metrics(y_true, y_pred)
    r2 = compute_metrics(y_pred, y_true)
    assert r1["SMAPE"] == pytest.approx(r2["SMAPE"])


def test_mae_is_symmetric():
    y_true = np.array([1.0, 3.0])
    y_pred = np.array([2.0, 1.0])
    r1 = compute_metrics(y_true, y_pred)
    r2 = compute_metrics(y_pred, y_true)
    assert r1["MAE"] == pytest.approx(r2["MAE"])


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_all_metrics_are_floats():
    result = compute_metrics([1.0, 2.0], [1.5, 2.5])
    for key in ("MAE", "RMSE", "MAPE", "SMAPE"):
        assert isinstance(result[key], float), f"{key} should be a float"


def test_returns_all_four_keys():
    result = compute_metrics([1.0], [2.0])
    assert set(result.keys()) == {"MAE", "RMSE", "MAPE", "SMAPE"}
