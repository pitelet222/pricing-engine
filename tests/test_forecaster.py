"""
Unit tests for src/models/forecaster.py.

Only the pure helper functions (build_nixtla_df, build_future_exog,
compute_ensemble_weights) are tested here. The heavy ML functions
(run_statistical_forecast, run_neural_forecast) require statsforecast /
neuralforecast and are covered by integration tests instead.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.forecaster import H, build_future_exog, build_nixtla_df, compute_ensemble_weights

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_feature_df(
    n_weeks: int = 20,
    regions: list | None = None,
) -> pd.DataFrame:
    if regions is None:
        regions = ["Albany"]
    rows = []
    dates = pd.date_range("2016-01-03", periods=n_weeks, freq="W")
    for region in regions:
        for is_organic in (0, 1):
            for d in dates:
                rows.append({
                    "Date": d,
                    "AveragePrice": 1.50,
                    "region": region,
                    "is_organic": is_organic,
                    "month_sin": np.sin(2 * np.pi * d.month / 12),
                    "month_cos": np.cos(2 * np.pi * d.month / 12),
                    "week_sin": np.sin(2 * np.pi * d.isocalendar()[1] / 52),
                    "week_cos": np.cos(2 * np.pi * d.isocalendar()[1] / 52),
                    "is_holiday_week": 0,
                    "region_price_level": 1.5,
                    "region_volume_level": 10.0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_nixtla_df
# ---------------------------------------------------------------------------

def test_build_nixtla_df_unique_id_format():
    df = _make_feature_df(regions=["Albany"])
    nf = build_nixtla_df(df)
    expected_ids = {"Albany_conventional", "Albany_organic"}
    assert set(nf["unique_id"].unique()) == expected_ids


def test_build_nixtla_df_renames_date_to_ds():
    df = _make_feature_df()
    nf = build_nixtla_df(df)
    assert "ds" in nf.columns
    assert "Date" not in nf.columns


def test_build_nixtla_df_renames_price_to_y():
    df = _make_feature_df()
    nf = build_nixtla_df(df)
    assert "y" in nf.columns
    assert "AveragePrice" not in nf.columns


def test_build_nixtla_df_contains_futr_exog():
    df = _make_feature_df()
    nf = build_nixtla_df(df)
    for col in ("month_sin", "month_cos", "week_sin", "week_cos", "is_holiday_week"):
        assert col in nf.columns


def test_build_nixtla_df_sorted_by_unique_id_and_ds():
    df = _make_feature_df(regions=["Houston", "Albany"])
    nf = build_nixtla_df(df)
    prev_uid, prev_ds = None, None
    for uid, ds in zip(nf["unique_id"], nf["ds"]):
        if prev_uid == uid:
            assert ds >= prev_ds
        prev_uid, prev_ds = uid, ds


# ---------------------------------------------------------------------------
# build_future_exog
# ---------------------------------------------------------------------------

def test_build_future_exog_exact_h_rows_per_series():
    df = _make_feature_df(regions=["Albany", "Houston"])
    nf = build_nixtla_df(df)
    futr = build_future_exog(nf, h=H)
    counts = futr.groupby("unique_id").size()
    assert (counts == H).all(), f"Expected {H} rows per series, got {counts.to_dict()}"


def test_build_future_exog_dates_strictly_after_training():
    df = _make_feature_df(regions=["Albany"])
    nf = build_nixtla_df(df)
    last_dates = nf.groupby("unique_id")["ds"].max()
    futr = build_future_exog(nf, h=H)
    for uid, last_date in last_dates.items():
        future_rows = futr[futr["unique_id"] == uid]
        assert (future_rows["ds"] > last_date).all()


def test_build_future_exog_columns():
    df = _make_feature_df()
    nf = build_nixtla_df(df)
    futr = build_future_exog(nf, h=H)
    expected = {"unique_id", "ds", "month_sin", "month_cos", "week_sin", "week_cos", "is_holiday_week"}
    assert expected.issubset(set(futr.columns))


def test_build_future_exog_is_holiday_week_zero():
    df = _make_feature_df()
    nf = build_nixtla_df(df)
    futr = build_future_exog(nf, h=H)
    assert (futr["is_holiday_week"] == 0).all()


def test_build_future_exog_cyclical_in_range():
    df = _make_feature_df()
    nf = build_nixtla_df(df)
    futr = build_future_exog(nf, h=H)
    for col in ("month_sin", "month_cos", "week_sin", "week_cos"):
        assert futr[col].between(-1.0, 1.0).all()


# ---------------------------------------------------------------------------
# compute_ensemble_weights
# ---------------------------------------------------------------------------

def test_compute_ensemble_weights_sum_to_one():
    all_cv = pd.DataFrame({
        "unique_id": ["A"] * 10,
        "y": np.ones(10),
        "ModelA": np.ones(10) * 1.1,
        "ModelB": np.ones(10) * 1.2,
    })
    weights = compute_ensemble_weights(all_cv, ["ModelA", "ModelB"])
    assert sum(weights.values()) == pytest.approx(1.0)


def test_compute_ensemble_weights_better_model_gets_higher_weight():
    all_cv = pd.DataFrame({
        "y": np.ones(10),
        "GoodModel": np.ones(10) * 1.05,   # MAE = 0.05
        "BadModel": np.ones(10) * 1.50,    # MAE = 0.50
    })
    weights = compute_ensemble_weights(all_cv, ["GoodModel", "BadModel"])
    assert weights["GoodModel"] > weights["BadModel"]


def test_compute_ensemble_weights_all_positive():
    all_cv = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "M1": [1.1, 2.2, 3.3],
        "M2": [0.9, 1.8, 2.7],
        "M3": [1.5, 2.0, 2.5],
    })
    weights = compute_ensemble_weights(all_cv, ["M1", "M2", "M3"])
    for name, w in weights.items():
        assert w > 0, f"Weight for {name} should be positive"


def test_compute_ensemble_weights_returns_all_models():
    all_cv = pd.DataFrame({
        "y": [1.0, 2.0],
        "ModelX": [1.1, 2.1],
        "ModelY": [1.2, 2.2],
    })
    models = ["ModelX", "ModelY"]
    weights = compute_ensemble_weights(all_cv, models)
    assert set(weights.keys()) == set(models)
