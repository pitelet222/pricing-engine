"""
Unit tests for src/data/preprocessing.py.

All tests are self-contained — no file I/O and no heavy dependencies.
clean_data is tested against in-memory DataFrames that mimic the raw
avocado CSV schema.
"""

import pathlib
import tempfile

import pandas as pd

from src.data.preprocessing import AGGREGATE_REGIONS, clean_data, load_raw_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_df(regions: list[str], n_dates: int = 3) -> pd.DataFrame:
    """Minimal DataFrame matching the post-load_raw_data schema."""
    dates = pd.date_range("2015-01-01", periods=n_dates, freq="W")
    rows = []
    for region in regions:
        for d in dates:
            for avocado_type in ("organic", "conventional"):
                rows.append(
                    {
                        "Date": d,
                        "AveragePrice": 1.5,
                        "Total Volume": 1000.0,
                        "4046": 400.0,
                        "4225": 350.0,
                        "4770": 50.0,
                        "Total Bags": 200.0,
                        "Small Bags": 150.0,
                        "Large Bags": 40.0,
                        "XLarge Bags": 10.0,
                        "type": avocado_type,
                        "year": d.year,
                        "region": region,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AGGREGATE_REGIONS constant
# ---------------------------------------------------------------------------


def test_aggregate_regions_contains_totalus():
    assert "TotalUS" in AGGREGATE_REGIONS


def test_aggregate_regions_contains_all_expected():
    expected = {
        "TotalUS",
        "West",
        "SouthCentral",
        "Southeast",
        "Northeast",
        "Midsouth",
        "GreatLakes",
        "Plains",
        "California",
        "WestTexNewMexico",
        "NorthernNewEngland",
    }
    assert expected == set(AGGREGATE_REGIONS)


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------


def test_clean_data_removes_aggregate_regions():
    df = _make_raw_df(["TotalUS", "West", "Albany"])
    cleaned = clean_data(df)
    assert "TotalUS" not in cleaned["region"].values
    assert "West" not in cleaned["region"].values


def test_clean_data_keeps_city_regions():
    df = _make_raw_df(["Albany", "Houston"])
    cleaned = clean_data(df)
    assert set(cleaned["region"].unique()) == {"Albany", "Houston"}


def test_clean_data_encodes_organic_as_one():
    df = _make_raw_df(["Albany"])
    cleaned = clean_data(df)
    assert set(cleaned["is_organic"].unique()) == {0, 1}
    organic_rows = cleaned[cleaned["is_organic"] == 1]
    assert len(organic_rows) > 0


def test_clean_data_conventional_encodes_as_zero():
    df = _make_raw_df(["Albany"])
    cleaned = clean_data(df)
    assert cleaned["is_organic"].dtype in (int, "int64", "int32")


def test_clean_data_drops_type_column():
    df = _make_raw_df(["Albany"])
    cleaned = clean_data(df)
    assert "type" not in cleaned.columns


def test_clean_data_drops_year_column():
    df = _make_raw_df(["Albany"])
    cleaned = clean_data(df)
    assert "year" not in cleaned.columns


def test_clean_data_resets_index():
    df = _make_raw_df(["Albany", "TotalUS"])
    cleaned = clean_data(df)
    assert list(cleaned.index) == list(range(len(cleaned)))


def test_clean_data_all_aggregate_regions_removed():
    all_regions = list(AGGREGATE_REGIONS) + ["Albany", "Houston"]
    df = _make_raw_df(all_regions)
    cleaned = clean_data(df)
    remaining = set(cleaned["region"].unique())
    assert remaining.isdisjoint(AGGREGATE_REGIONS)


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------


def test_load_raw_data_parses_date():
    raw = _make_raw_df(["Albany"], n_dates=3)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        raw.to_csv(f.name)
        path = pathlib.Path(f.name)
    try:
        loaded = load_raw_data(path)
        assert pd.api.types.is_datetime64_any_dtype(loaded["Date"])
    finally:
        path.unlink(missing_ok=True)


def test_load_raw_data_sorted_ascending():
    raw = _make_raw_df(["Albany"], n_dates=5)
    raw = raw.sample(frac=1, random_state=0)  # shuffle
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        raw.to_csv(f.name)
        path = pathlib.Path(f.name)
    try:
        loaded = load_raw_data(path)
        dates = loaded["Date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
    finally:
        path.unlink(missing_ok=True)
