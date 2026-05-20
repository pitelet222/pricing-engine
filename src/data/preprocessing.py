"""
preprocessing.py — Raw data loading and schema normalisation.

Extracted from notebooks/02_feature_engineering.ipynb (sections 2-3).
Provides the first stage of the data pipeline: load the raw avocado CSV,
filter out aggregate/macro regions, and encode the avocado type flag.

Functions
---------
load_raw_data  : Load avocado.csv, parse dates, sort by date.
clean_data     : Filter aggregate regions, encode is_organic, drop redundant columns.
"""

from __future__ import annotations

import pathlib

import pandas as pd

# Aggregate / macro regions to exclude from analysis.
# The Hass Avocado Board dataset mixes 43 city-level markets with 11 aggregations
# (e.g. TotalUS = sum of all US regions). Keeping aggregates would double-count
# volume and distort per-region demand models.
AGGREGATE_REGIONS: frozenset[str] = frozenset({
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
})


def load_raw_data(path: pathlib.Path | str) -> pd.DataFrame:
    """
    Load the raw avocado CSV and normalise the date column.

    The raw file uses an unnamed integer index column (col 0) which is
    dropped via index_col=0. Date is parsed to datetime and the DataFrame
    is sorted chronologically so downstream feature engineering can rely
    on temporal ordering.

    Parameters
    ----------
    path : Path to avocado.csv (data/raw/avocado.csv).

    Returns
    -------
    DataFrame with 18 249 rows × 13 columns, sorted by Date.
    """
    df = pd.read_csv(path, index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove aggregate regions, encode avocado type, and drop redundant columns.

    Steps
    -----
    1. Remove rows where region is in AGGREGATE_REGIONS (macro-level aggregations).
       Leaves 43 city-level regions and 14 534 rows.
    2. Encode the 'type' column ('organic'/'conventional') as a binary integer
       'is_organic' flag (1 = organic, 0 = conventional).
    3. Drop 'type' (replaced by is_organic) and 'year' (redundant with Date —
       richer temporal features are added by features.add_temporal_features).

    Parameters
    ----------
    df : DataFrame from load_raw_data.

    Returns
    -------
    DataFrame with 14 534 rows and columns:
    Date, AveragePrice, Total Volume, 4046, 4225, 4770,
    Total Bags, Small Bags, Large Bags, XLarge Bags, region, is_organic.
    """
    df = df[~df["region"].isin(AGGREGATE_REGIONS)].copy()
    df["is_organic"] = (df["type"] == "organic").astype(int)
    df = df.drop(columns=["type", "year"])
    return df.reset_index(drop=True)
