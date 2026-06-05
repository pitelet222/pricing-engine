"""
Unit tests for src/evaluation/explainability.py.

build_top_drivers and generate_pricing_narrative are tested with a
_MockExplanation class that matches the shap.Explanation interface used
by those functions — no shap package import needed in CI.
"""

import numpy as np
import pandas as pd

from src.evaluation.explainability import build_top_drivers, generate_pricing_narrative

# ---------------------------------------------------------------------------
# Mock SHAP explanation
# ---------------------------------------------------------------------------


class _MockExplanation:
    """Minimal shap.Explanation stand-in used by build_top_drivers."""

    def __init__(self, values: np.ndarray, base_values: np.ndarray):
        self.values = values
        self.base_values = base_values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_COLS = ["AveragePrice", "is_organic", "region_encoded", "price_lag_1w", "month_sin"]
N_FEATURES = len(FEATURE_COLS)


def _make_latest_ctx(n_series: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_series):
        row = {col: rng.uniform(0, 2) for col in FEATURE_COLS}
        row["unique_id"] = f"Series_{i}"
        rows.append(row)
    return pd.DataFrame(rows)


def _make_shap_explanation(n_series: int = 4) -> _MockExplanation:
    rng = np.random.default_rng(1)
    values = rng.normal(0, 0.5, (n_series, N_FEATURES))
    base_values = np.full(n_series, 10.85)
    return _MockExplanation(values, base_values)


def _make_recs_df(n_series: int = 4, uplift_pct: float = 5.0) -> pd.DataFrame:
    rows = []
    for i in range(n_series):
        rows.append(
            {
                "unique_id": f"Series_{i}",
                "current_price": 1.50,
                "optimal_price": 1.50 * (1 + uplift_pct / 100),
                "revenue_change_pct": uplift_pct,
                "price_change_pct": uplift_pct,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_top_drivers
# ---------------------------------------------------------------------------


def test_build_top_drivers_output_columns():
    ctx = _make_latest_ctx()
    shap_ex = _make_shap_explanation()
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    expected = {
        "unique_id",
        "driver_rank",
        "feature",
        "shap_value",
        "abs_shap_value",
        "direction",
        "feature_value",
    }
    assert expected.issubset(set(result.columns))


def test_build_top_drivers_row_count():
    n_series, n_drivers = 4, 3
    ctx = _make_latest_ctx(n_series)
    shap_ex = _make_shap_explanation(n_series)
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=n_drivers)
    assert len(result) == n_series * n_drivers


def test_build_top_drivers_driver_ranks():
    ctx = _make_latest_ctx()
    shap_ex = _make_shap_explanation()
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    assert set(result["driver_rank"].unique()) == {1, 2, 3}


def test_build_top_drivers_valid_direction_values():
    ctx = _make_latest_ctx()
    shap_ex = _make_shap_explanation()
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    valid = {"increases_demand", "decreases_demand"}
    assert set(result["direction"].unique()).issubset(valid)


def test_build_top_drivers_direction_matches_sign():
    ctx = _make_latest_ctx(n_series=1)
    # Force known SHAP values: first feature positive, second negative
    values = np.array([[1.0, -0.5, 0.3, 0.1, 0.2]])
    shap_ex = _MockExplanation(values=values, base_values=np.array([10.85]))
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    # rank 1 = largest abs = feature 0 (shap=+1.0) → increases_demand
    rank1 = result[result["driver_rank"] == 1].iloc[0]
    assert rank1["feature"] == FEATURE_COLS[0]
    assert rank1["direction"] == "increases_demand"


def test_build_top_drivers_abs_shap_nonnegative():
    ctx = _make_latest_ctx()
    shap_ex = _make_shap_explanation()
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    assert (result["abs_shap_value"] >= 0).all()


def test_build_top_drivers_top_ranked_has_largest_abs_shap():
    ctx = _make_latest_ctx()
    shap_ex = _make_shap_explanation()
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    for uid in result["unique_id"].unique():
        group = result[result["unique_id"] == uid].sort_values("driver_rank")
        abs_vals = group["abs_shap_value"].values
        assert abs_vals[0] >= abs_vals[1]
        assert abs_vals[1] >= abs_vals[2]


def test_build_top_drivers_feature_in_feature_cols():
    ctx = _make_latest_ctx()
    shap_ex = _make_shap_explanation()
    result = build_top_drivers(shap_ex, ctx, FEATURE_COLS, n=3)
    assert result["feature"].isin(FEATURE_COLS).all()


# ---------------------------------------------------------------------------
# generate_pricing_narrative
# ---------------------------------------------------------------------------


def test_generate_pricing_narrative_contains_uid():
    uid = "Albany_organic"
    shap_row = np.array([0.1, -0.3, 0.5, -0.2, 0.05])
    recs = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "current_price": 1.50,
                "optimal_price": 1.80,
                "revenue_change_pct": 5.0,
                "price_change_pct": 20.0,
            }
        ]
    )
    narrative = generate_pricing_narrative(uid, shap_row, FEATURE_COLS, 10.85, recs)
    assert uid in narrative


def test_generate_pricing_narrative_raise_action():
    uid = "Albany_organic"
    shap_row = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    recs = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "current_price": 1.50,
                "optimal_price": 1.80,  # raise
                "revenue_change_pct": 8.0,
                "price_change_pct": 20.0,
            }
        ]
    )
    narrative = generate_pricing_narrative(uid, shap_row, FEATURE_COLS, 10.85, recs)
    assert "raise" in narrative.lower()
    assert "cut" not in narrative.lower()


def test_generate_pricing_narrative_cut_action():
    uid = "Albany_organic"
    shap_row = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    recs = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "current_price": 1.50,
                "optimal_price": 1.20,  # cut
                "revenue_change_pct": 3.0,
                "price_change_pct": -20.0,
            }
        ]
    )
    narrative = generate_pricing_narrative(uid, shap_row, FEATURE_COLS, 10.85, recs)
    assert "cut" in narrative.lower()
    assert "raise" not in narrative.lower()


def test_generate_pricing_narrative_returns_string():
    uid = "Albany_organic"
    shap_row = np.zeros(N_FEATURES)
    recs = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "current_price": 1.50,
                "optimal_price": 1.60,
                "revenue_change_pct": 2.0,
                "price_change_pct": 6.7,
            }
        ]
    )
    result = generate_pricing_narrative(uid, shap_row, FEATURE_COLS, 10.85, recs)
    assert isinstance(result, str)


def test_generate_pricing_narrative_contains_top_features():
    uid = "TestSeries"
    shap_row = np.array([1.5, -0.8, 0.3, 0.1, 0.05])  # AveragePrice top driver
    recs = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "current_price": 1.50,
                "optimal_price": 1.70,
                "revenue_change_pct": 4.0,
                "price_change_pct": 13.3,
            }
        ]
    )
    narrative = generate_pricing_narrative(uid, shap_row, FEATURE_COLS, 10.85, recs, top_n=2)
    # Top 2 drivers by |SHAP|: AveragePrice (1.5) and is_organic (-0.8)
    assert "AveragePrice" in narrative
    assert "is_organic" in narrative


def test_generate_pricing_narrative_uplift_included():
    uid = "TestSeries"
    shap_row = np.array([0.2, 0.3, 0.1, 0.4, 0.05])
    recs = pd.DataFrame(
        [
            {
                "unique_id": uid,
                "current_price": 1.50,
                "optimal_price": 1.65,
                "revenue_change_pct": 12.5,
                "price_change_pct": 10.0,
            }
        ]
    )
    narrative = generate_pricing_narrative(uid, shap_row, FEATURE_COLS, 10.85, recs)
    assert "12.5" in narrative
