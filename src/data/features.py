"""
features.py — Feature engineering for the avocado demand model.

Extracted from notebooks/02_feature_engineering.ipynb (sections 4-9).
Transforms a cleaned avocado DataFrame (output of preprocessing.clean_data)
into a 42-column feature matrix matching data/processed/avocado_features.csv.

Feature groups
--------------
Temporal    : month, week, quarter, sin/cos cyclical encodings, holiday flag
Volume/mix  : log1p transforms, PLU share ratios, bag share and size ratio
Lag/rolling : per-(region, is_organic) group lags and rolling statistics
Region      : target-encoded price and volume levels (train era ≤ 2017 only)

Functions
---------
add_temporal_features   : Calendar and cyclical time features.
add_holiday_flag        : Binary flag for US avocado-demand holiday weeks.
add_volume_features     : Log transforms, PLU shares, bag features.
add_lag_rolling_features: Price lags, rolling mean/std, momentum per group.
add_region_encodings    : Target-encoded region_price_level and region_volume_level.
finalize_features       : Drop NaN warm-up rows and raw replaced columns.
build_features          : Run the complete pipeline in order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOLUME_COLS: list[str] = [
    "Total Volume",
    "4046",
    "4225",
    "4770",
    "Total Bags",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
]
PLU_CODES: list[str] = ["4046", "4225", "4770"]

# Columns replaced by engineered features; dropped in finalize_features.
# Raw PLU and bag volumes → log transforms + share ratios.
# 'season' string label → cyclical month/week encodings.
DROP_COLS: list[str] = [
    "4046",
    "4225",
    "4770",
    "Total Bags",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
    "season",
]

_GROUP_COLS: list[str] = ["region", "is_organic"]
_TRAIN_CUTOFF_YEAR: int = 2017  # target-encode regions on train era only


def _build_holiday_weeks() -> frozenset[tuple[int, int]]:
    """
    Pre-compute (year, ISO-week) pairs for major US avocado-demand holidays, 2015-2018.

    Mirrors notebook 02: US federal holidays + Super Bowl + Cinco de Mayo.
    Uses only pandas (no external holidays package) so it works in all environments.
    """
    dates = {
        # New Year's Day
        "2015-01-01",
        "2016-01-01",
        "2017-01-02",
        "2018-01-01",
        # Super Bowl (first Sunday of February)
        "2015-02-01",
        "2016-02-07",
        "2017-02-05",
        "2018-02-04",
        # Cinco de Mayo
        "2015-05-05",
        "2016-05-05",
        "2017-05-05",
        "2018-05-05",
        # Independence Day
        "2015-07-04",
        "2016-07-04",
        "2017-07-04",
        "2018-07-04",
        # Labor Day (first Monday of September)
        "2015-09-07",
        "2016-09-05",
        "2017-09-04",
        "2018-09-03",
        # Thanksgiving (fourth Thursday of November)
        "2015-11-26",
        "2016-11-24",
        "2017-11-23",
        "2018-11-22",
        # Christmas Day
        "2015-12-25",
        "2016-12-25",
        "2017-12-25",
        "2018-12-25",
    }
    result: set[tuple[int, int]] = set()
    for d in dates:
        iso = pd.Timestamp(d).isocalendar()
        result.add((iso[0], iso[1]))
    return frozenset(result)


# Pre-computed at module import — O(1) lookup in add_holiday_flag.
_HOLIDAY_WEEKS: frozenset[tuple[int, int]] = _build_holiday_weeks()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and cyclical time features derived from the Date column.

    Cyclical encodings (sin/cos) preserve circular distance so that
    week 52 and week 1 are treated as neighbours, not 51 apart.

    Columns added
    -------------
    year, month, week, day_of_week, quarter, season (dropped later),
    month_sin, month_cos, week_sin, week_cos.
    """
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["quarter"] = df["Date"].dt.quarter

    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
    }
    df["season"] = df["month"].map(season_map)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an is_holiday_week flag for weeks containing major US avocado-demand events.

    Requires the 'year' and 'week' columns added by add_temporal_features.
    Uses _HOLIDAY_WEEKS (pre-computed from known 2015-2018 dates) for O(1) lookup.

    Columns added
    -------------
    is_holiday_week : int (1 = holiday week, 0 = non-holiday week).
    """
    df = df.copy()
    df["is_holiday_week"] = df.apply(
        lambda r: (int(r["year"]), int(r["week"])) in _HOLIDAY_WEEKS, axis=1
    ).astype(int)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed volume columns and mix-ratio features.

    Log transforms (log1p) reduce right-skew in volume counts. PLU share
    ratios capture consumer size preferences independently of scale. Bag
    features summarise the bag dimension without keeping redundant correlated
    raw columns.

    Columns added
    -------------
    log_total_volume, log_4046, log_4225, log_4770, log_total_bags,
    log_small_bags, log_large_bags, log_xlarge_bags,
    plu_share_4046, plu_share_4225, plu_share_4770,
    bag_share, bag_size_ratio.
    """
    df = df.copy()

    for col in VOLUME_COLS:
        df[f"log_{col.replace(' ', '_').lower()}"] = np.log1p(df[col])

    for code in PLU_CODES:
        df[f"plu_share_{code}"] = df[code] / df["Total Volume"].replace(0, np.nan)
        df[f"plu_share_{code}"] = df[f"plu_share_{code}"].fillna(0)

    df["bag_share"] = df["Total Bags"] / df["Total Volume"].replace(0, np.nan)
    df["bag_share"] = df["bag_share"].fillna(0).clip(upper=1.0)
    # log-ratio: Large Bags preference vs Small Bags, reducing skew
    df["bag_size_ratio"] = np.log1p(df["Large Bags"]) - np.log1p(df["Small Bags"])
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price lag, rolling mean/std, and momentum features per (region, is_organic) group.

    shift(1) is applied inside rolling windows to prevent look-ahead bias:
    the feature at time t uses only observations up to t-1. Groups ensure
    lags never cross region or type boundaries.

    Columns added
    -------------
    price_lag_1w, price_lag_2w, price_lag_4w,
    volume_lag_1w, volume_lag_4w,
    price_rmean_4w, price_rmean_8w, price_rmean_12w,
    price_rstd_4w,
    price_pct_change_1w, price_momentum_4w,
    volume_rmean_4w.
    """
    df = df.sort_values([*_GROUP_COLS, "Date"]).reset_index(drop=True)

    for lag in [1, 2, 4]:
        df[f"price_lag_{lag}w"] = df.groupby(_GROUP_COLS)["AveragePrice"].shift(lag)

    for lag in [1, 4]:
        df[f"volume_lag_{lag}w"] = df.groupby(_GROUP_COLS)["Total Volume"].shift(lag)

    for window in [4, 8, 12]:
        df[f"price_rmean_{window}w"] = df.groupby(_GROUP_COLS)["AveragePrice"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    df["price_rstd_4w"] = df.groupby(_GROUP_COLS)["AveragePrice"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )

    df["price_pct_change_1w"] = df.groupby(_GROUP_COLS)["AveragePrice"].pct_change(1)
    df["price_momentum_4w"] = df["AveragePrice"] - df["price_rmean_4w"]

    df["volume_rmean_4w"] = df.groupby(_GROUP_COLS)["Total Volume"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return df


def add_region_encodings(
    df: pd.DataFrame,
    train_cutoff_year: int = _TRAIN_CUTOFF_YEAR,
) -> pd.DataFrame:
    """
    Add target-encoded region features computed on train-era data only.

    Computing encodings on train data (year ≤ train_cutoff_year) prevents
    leakage from future price observations into the region-level features.

    Columns added
    -------------
    region_price_level  : Mean AveragePrice per region (train era).
    region_volume_level : log1p(mean Total Volume per region) (train era).
    """
    df = df.copy()
    train_mask = df["year"] <= train_cutoff_year

    region_price_mean = (
        df.loc[train_mask].groupby("region")["AveragePrice"].mean().rename("region_price_level")
    )
    region_vol_mean = (
        df.loc[train_mask].groupby("region")["Total Volume"].mean().rename("region_volume_level")
    )

    df = df.merge(region_price_mean, on="region", how="left")
    df = df.merge(region_vol_mean, on="region", how="left")
    df["region_volume_level"] = np.log1p(df["region_volume_level"])
    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop warm-up NaN rows and raw columns replaced by engineered alternatives.

    NaN rows arise from the 4-week warm-up period of lag features at the
    start of each (region, is_organic) group — dropna on the most restrictive
    lag/rolling columns removes them all in one pass.

    Dropped columns (DROP_COLS)
    ---------------------------
    Raw PLU volumes (4046, 4225, 4770) and bag volumes (Total Bags, Small Bags,
    Large Bags, XLarge Bags) are replaced by log transforms and share ratios.
    The 'season' string label is replaced by cyclical month/week encodings.
    """
    df = df.dropna(subset=["price_lag_4w", "price_rmean_12w", "price_rstd_4w"]).copy()
    return df.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)


def build_features(
    raw_df: pd.DataFrame,
    train_cutoff_year: int = _TRAIN_CUTOFF_YEAR,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline on a cleaned avocado DataFrame.

    Input
    -----
    raw_df : DataFrame from preprocessing.clean_data — must contain:
             Date, AveragePrice, Total Volume, 4046, 4225, 4770,
             Total Bags, Small Bags, Large Bags, XLarge Bags,
             region, is_organic.

    Output
    ------
    DataFrame with 42 columns matching data/processed/avocado_features.csv
    (14 190 rows × 42 columns for the full avocado dataset).
    """
    df = add_temporal_features(raw_df)
    df = add_holiday_flag(df)
    df = add_volume_features(df)
    df = add_lag_rolling_features(df)
    df = add_region_encodings(df, train_cutoff_year=train_cutoff_year)
    return finalize_features(df)
