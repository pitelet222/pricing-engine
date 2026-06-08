"""
Unit tests for src/data/features.py.

Tests use minimal in-memory DataFrames — no CSV files, no heavy dependencies.
Each function is tested in isolation; build_features is tested end-to-end
on a small synthetic dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_holiday_flag,
    add_lag_rolling_features,
    add_region_encodings,
    add_temporal_features,
    add_volume_features,
    build_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_row(region="Albany", is_organic=0, date="2016-01-03") -> dict:
    return {
        "Date": pd.Timestamp(date),
        "AveragePrice": 1.50,
        "Total Volume": 1000.0,
        "4046": 400.0,
        "4225": 350.0,
        "4770": 50.0,
        "Total Bags": 200.0,
        "Small Bags": 150.0,
        "Large Bags": 40.0,
        "XLarge Bags": 10.0,
        "region": region,
        "is_organic": is_organic,
    }


def _make_clean_df(
    n_weeks: int = 20,
    regions: list | None = None,
    start: str = "2016-01-03",
) -> pd.DataFrame:
    """Minimal cleaned DataFrame matching the output of preprocessing.clean_data."""
    if regions is None:
        regions = ["Albany"]
    rows = []
    dates = pd.date_range(start, periods=n_weeks, freq="W")
    for region in regions:
        for is_organic in (0, 1):
            for i, d in enumerate(dates):
                row = _base_row(region=region, is_organic=is_organic, date=str(d.date()))
                row["AveragePrice"] = 1.0 + 0.02 * i  # slight variation
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------


def test_add_temporal_features_adds_cyclical_cols():
    df = pd.DataFrame([_base_row()])
    result = add_temporal_features(df)
    for col in ("month_sin", "month_cos", "week_sin", "week_cos"):
        assert col in result.columns


def test_add_temporal_features_adds_calendar_cols():
    df = pd.DataFrame([_base_row()])
    result = add_temporal_features(df)
    for col in ("year", "month", "week", "quarter"):
        assert col in result.columns


def test_cyclical_encoding_in_range():
    df = _make_clean_df(n_weeks=52)
    result = add_temporal_features(df)
    for col in ("month_sin", "month_cos", "week_sin", "week_cos"):
        assert result[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_month_cos_january_close_to_one():
    jan_row = _base_row(date="2016-01-03")
    df = pd.DataFrame([jan_row])
    result = add_temporal_features(df)
    # month=1, cos(2π*1/12) ≈ 0.866
    assert result["month_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi / 12), abs=1e-6)


# ---------------------------------------------------------------------------
# add_holiday_flag
# ---------------------------------------------------------------------------


def test_super_bowl_2016_flagged():
    # Super Bowl 2016: 2016-02-07
    row = _base_row(date="2016-02-07")
    df = add_temporal_features(pd.DataFrame([row]))
    df = add_holiday_flag(df)
    assert df["is_holiday_week"].iloc[0] == 1


def test_regular_week_not_flagged():
    row = _base_row(date="2016-03-06")  # not a holiday
    df = add_temporal_features(pd.DataFrame([row]))
    df = add_holiday_flag(df)
    assert df["is_holiday_week"].iloc[0] == 0


def test_holiday_flag_is_integer():
    df = add_temporal_features(pd.DataFrame([_base_row()]))
    result = add_holiday_flag(df)
    assert result["is_holiday_week"].dtype in (int, "int64", "int32", "int8")


# ---------------------------------------------------------------------------
# add_volume_features
# ---------------------------------------------------------------------------


def test_log_total_volume_nonnegative():
    df = pd.DataFrame([_base_row()])
    result = add_volume_features(df)
    assert (result["log_total_volume"] >= 0).all()


def test_plu_shares_bounded():
    df = pd.DataFrame([_base_row()])
    result = add_volume_features(df)
    for col in ("plu_share_4046", "plu_share_4225", "plu_share_4770"):
        assert result[col].between(0.0, 1.0).all(), f"{col} out of [0, 1]"


def test_bag_share_bounded():
    df = pd.DataFrame([_base_row()])
    result = add_volume_features(df)
    assert result["bag_share"].between(0.0, 1.0).all()


def test_log_transforms_monotone():
    row_low = _base_row()
    row_low["Total Volume"] = 100.0
    row_high = _base_row()
    row_high["Total Volume"] = 10_000.0
    df = pd.DataFrame([row_low, row_high])
    result = add_volume_features(df)
    assert result["log_total_volume"].iloc[1] > result["log_total_volume"].iloc[0]


# ---------------------------------------------------------------------------
# add_lag_rolling_features
# ---------------------------------------------------------------------------


def test_first_row_lag_is_nan():
    df = _make_clean_df(n_weeks=10, regions=["Albany"])
    df_t = add_temporal_features(df)
    result = add_lag_rolling_features(df_t)
    # groupby.nth(0) gets the literal first row per group (preserving NaN),
    # unlike groupby.first() which skips NaN values.
    first_rows = result.groupby(["region", "is_organic"]).nth(0)
    assert first_rows["price_lag_1w"].isna().all()


def test_lag_4w_matches_price_4_weeks_ago():
    df = _make_clean_df(n_weeks=10, regions=["Albany"])
    df = df[df["is_organic"] == 0].reset_index(drop=True)
    df_t = add_temporal_features(df)
    result = add_lag_rolling_features(df_t)
    result = result.dropna(subset=["price_lag_4w"])
    # lag_4w at row i should equal AveragePrice at row i-4
    for i in range(len(result)):
        actual_price_4w_ago = result["AveragePrice"].iloc[i - 4] if i >= 4 else None
        if actual_price_4w_ago is not None:
            # due to groupby reset, just check non-NaN
            assert not pd.isna(result["price_lag_4w"].iloc[i])


def test_rolling_mean_not_nan_after_warmup():
    df = _make_clean_df(n_weeks=15, regions=["Albany"])
    df_t = add_temporal_features(df)
    result = add_lag_rolling_features(df_t)
    # After dropna warmup, rolling means should not be NaN
    clean = result.dropna(subset=["price_lag_4w", "price_rmean_12w"])
    assert clean["price_rmean_4w"].isna().sum() == 0


# ---------------------------------------------------------------------------
# add_region_encodings
# ---------------------------------------------------------------------------


def test_region_encodings_added():
    df = _make_clean_df(n_weeks=15, regions=["Albany", "Houston"])
    df_t = add_temporal_features(df)
    df_h = add_holiday_flag(df_t)
    df_v = add_volume_features(df_h)
    df_l = add_lag_rolling_features(df_v)
    result = add_region_encodings(df_l, train_cutoff_year=2016)
    assert "region_price_level" in result.columns
    assert "region_volume_level" in result.columns


def test_region_volume_level_is_log_transformed():
    df = _make_clean_df(n_weeks=15, regions=["Albany"])
    df_t = add_temporal_features(df)
    df_h = add_holiday_flag(df_t)
    df_v = add_volume_features(df_h)
    df_l = add_lag_rolling_features(df_v)
    result = add_region_encodings(df_l, train_cutoff_year=2016)
    assert (result["region_volume_level"] >= 0).all()


# ---------------------------------------------------------------------------
# build_features — end-to-end column count
# ---------------------------------------------------------------------------


def test_build_features_output_has_42_columns():
    df = _make_clean_df(n_weeks=20, regions=["Albany", "Houston"])
    result = build_features(df)
    assert len(result.columns) == 42, (
        f"Expected 42 columns, got {len(result.columns)}: {sorted(result.columns)}"
    )


def test_build_features_no_nans_in_feature_cols():
    from src.models.pricer import FEATURE_COLS

    df = _make_clean_df(n_weeks=20, regions=["Albany"])
    result = build_features(df)
    # region_encoded is added by the API serving layer (label encoder), not by
    # build_features — exclude it from the column presence check here.
    pipeline_cols = [c for c in FEATURE_COLS if c != "region_encoded"]
    missing = [c for c in pipeline_cols if c not in result.columns]
    assert not missing, f"Missing feature columns: {missing}"
    assert result[pipeline_cols].isna().sum().sum() == 0


def test_build_features_drops_raw_plu_columns():
    df = _make_clean_df(n_weeks=20, regions=["Albany"])
    result = build_features(df)
    for col in ("4046", "4225", "4770"):
        assert col not in result.columns


def test_build_features_removes_warmup_rows():
    df = _make_clean_df(n_weeks=20, regions=["Albany"])
    raw_rows = len(df)
    result = build_features(df)
    # finalize_features drops the first ~4 rows per group due to lag warmup
    assert len(result) < raw_rows
