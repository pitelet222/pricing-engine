"""
explainability.py — SHAP attribution and demand driver analysis.

Extracted from notebooks/06_explainability.ipynb.
Translates LightGBM demand model predictions into auditable, human-readable
explanations using SHAP (SHapley Additive exPlanations).  TreeExplainer gives
exact Shapley values for tree ensembles — every feature's contribution is a
provable fair-share allocation of the delta between the prediction and the
global average.

Two complementary SHAP sets are computed:

1. Global sample (stratified 3 000-row subset) — feature importance ranking,
   beeswarm / dependence plots, segment analysis (organic vs conventional).
2. Latest-context rows (one per series) — local waterfall plots and plain-English
   pricing narratives for the /explain API endpoint.

Constants
---------
SHAP_GLOBAL_SAMPLE : Target size for the stratified global SHAP sample.
SEED               : Random seed for reproducible stratified sampling.

Functions
---------
compute_shap_global  : Stratified sample + TreeExplainer on historical rows.
compute_shap_latest  : TreeExplainer on the latest-context rows (one per series).
build_top_drivers    : Top-n demand drivers per series from shap_latest.
generate_pricing_narrative : Plain-English markdown explanation per series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SHAP_GLOBAL_SAMPLE: int = 3_000
SEED: int = 42


def compute_shap_global(  # pragma: no cover
    model: Any,
    X_all: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_cols: list[str],
    n_sample: int = SHAP_GLOBAL_SAMPLE,
    seed: int = SEED,
) -> tuple[Any, pd.DataFrame]:
    """
    Compute global SHAP values on a stratified sample of historical rows.

    Stratification by unique_id ensures every series contributes proportionally
    to the global importance estimates — short and long series are treated fairly.

    The callable shap.Explanation API is used (explainer(X)) rather than
    explainer.shap_values(X) so the returned object is compatible with
    shap.plots.waterfall and shap.plots.beeswarm.

    Parameters
    ----------
    model       : Trained LightGBM Booster.
    X_all       : Full feature matrix (rows × feature_cols).
    meta_df     : DataFrame aligned to X_all with at least 'unique_id' column.
    feature_cols: Ordered list of feature names matching the model.
    n_sample    : Target global sample size (default 3 000).
    seed        : Random seed for reproducible stratified sampling.

    Returns
    -------
    (shap_global, meta_shap) where:
      shap_global : shap.Explanation with .values (n×p), .base_values, .data
      meta_shap   : Metadata DataFrame aligned row-wise to shap_global.
    """
    import shap

    sample_idx = (
        meta_df
        .groupby("unique_id", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), max(1, round(n_sample * len(g) / len(meta_df)))),
            random_state=seed,
        ))
        .index
    )
    X_shap = X_all.loc[sample_idx][feature_cols].reset_index(drop=True)
    meta_shap = meta_df.loc[sample_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_global = explainer(X_shap)
    return shap_global, meta_shap


def compute_shap_latest(  # pragma: no cover
    model: Any,
    X_latest: pd.DataFrame,
) -> Any:
    """
    Compute SHAP values for the latest-context rows (one per series).

    Uses TreeExplainer's callable API to return a shap.Explanation that bundles
    SHAP values, base values, and feature values in a single object.

    Parameters
    ----------
    model    : Trained LightGBM Booster.
    X_latest : Feature matrix for the most recent observation of each series.

    Returns
    -------
    shap.Explanation with .values shape (n_series × n_features).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer(X_latest)


def build_top_drivers(
    shap_latest: Any,
    latest_ctx: pd.DataFrame,
    feature_cols: list[str],
    n: int = 3,
) -> pd.DataFrame:
    """
    Extract the top-n demand drivers per series from the latest-context SHAP values.

    Each driver row records the feature name, its absolute and signed SHAP value,
    the direction of its effect (increases or decreases demand), and the actual
    feature value at the current context.  This is the exact payload returned
    by the /explain API endpoint.

    Parameters
    ----------
    shap_latest : shap.Explanation from compute_shap_latest (n_series × n_features).
    latest_ctx  : DataFrame with 'unique_id' and all feature_cols; row-aligned to
                  shap_latest.
    feature_cols: Ordered list of feature names matching shap_latest.values columns.
    n           : Number of top drivers per series (default 3).

    Returns
    -------
    DataFrame with columns:
    unique_id, driver_rank, feature, shap_value, abs_shap_value, direction,
    feature_value.
    """
    uid_to_idx = {uid: i for i, uid in enumerate(latest_ctx["unique_id"].values)}
    rows = []
    for uid, idx in uid_to_idx.items():
        shap_row = shap_latest.values[idx]
        abs_shap = np.abs(shap_row)
        top_idx = abs_shap.argsort()[-n:][::-1]
        feat_row = latest_ctx.iloc[idx]
        for rank, feat_idx in enumerate(top_idx, start=1):
            rows.append({
                "unique_id": uid,
                "driver_rank": rank,
                "feature": feature_cols[feat_idx],
                "shap_value": round(float(shap_row[feat_idx]), 6),
                "abs_shap_value": round(float(abs_shap[feat_idx]), 6),
                "direction": (
                    "increases_demand" if shap_row[feat_idx] > 0 else "decreases_demand"
                ),
                "feature_value": round(float(feat_row[feature_cols[feat_idx]]), 6),
            })
    return pd.DataFrame(rows)


def generate_pricing_narrative(
    uid: str,
    shap_row: np.ndarray,
    feature_names: list[str],
    base_val: float,
    recs_df: pd.DataFrame,
    top_n: int = 3,
) -> str:
    """
    Translate SHAP values at the current market context into a plain-English narrative.

    The returned markdown string is the payload for the /explain API endpoint:
    a human-readable justification for each pricing recommendation generated
    entirely from data — no hardcoded strings.

    Structure of the narrative
    --------------------------
    1. Recommendation headline: action (raise/cut) + expected revenue uplift.
    2. Demand attribution: global baseline → current prediction via top-n features.
    3. Pricing note: grid search bounds and direction of the recommended move.

    Parameters
    ----------
    uid          : Series identifier, e.g. 'LosAngeles_conventional'.
    shap_row     : SHAP values for one observation, shape (n_features,).
    feature_names: Ordered list of feature names aligned to shap_row.
    base_val     : Global baseline E[f(x)] in log-volume space.
    recs_df      : Pricing recommendations DataFrame; must contain columns:
                   unique_id, current_price, optimal_price,
                   revenue_change_pct, price_change_pct.
    top_n        : Number of top drivers to include in the narrative (default 3).

    Returns
    -------
    Markdown string ready to be rendered or returned as API text.
    """
    rec = recs_df[recs_df["unique_id"] == uid].iloc[0]
    current_price = rec["current_price"]
    optimal_price = rec["optimal_price"]
    revenue_uplift = rec["revenue_change_pct"]
    price_change = rec["price_change_pct"]

    top_idx = np.abs(shap_row).argsort()[-top_n:][::-1]
    driver_lines = []
    for rank, feat_idx in enumerate(top_idx, start=1):
        feat = feature_names[feat_idx]
        shap_val = shap_row[feat_idx]
        direction = "increases" if shap_val > 0 else "decreases"
        driver_lines.append(
            f"  {rank}. `{feat}` **{direction}** predicted demand "
            f"by {abs(shap_val):.4f} log-units  (SHAP = {shap_val:+.4f})"
        )

    if price_change < 0:
        action = (
            f"**cut price by {abs(price_change):.1f}%**"
            f"  (${current_price:.2f} -> ${optimal_price:.2f})"
        )
    else:
        action = (
            f"**raise price by {price_change:.1f}%**"
            f"  (${current_price:.2f} -> ${optimal_price:.2f})"
        )

    prediction = base_val + float(shap_row.sum())
    net_push = "above" if shap_row.sum() > 0 else "below"

    return (
        f"### {uid}\n\n"
        f"**Recommendation:** {action} -- "
        f"estimated revenue uplift: **+{revenue_uplift:.1f}%**\n\n"
        f"**Why does the model predict this demand level?**\n\n"
        f"Global baseline E[f(x)] = `{base_val:.4f}` "
        f"({np.exp(base_val):,.0f} avocados/week on average).  \n"
        f"The top {top_n} features at the current market context push the prediction "
        f"**{net_push}** that baseline to `{prediction:.4f}` "
        f"({np.exp(prediction):,.0f} avocados/week):\n\n"
        + "\n".join(driver_lines)
        + f"\n\n*Pricing note:* the revenue-maximising price was found via grid search "
        f"over ±30% of the historical mean. A {abs(price_change):.1f}% "
        f'{"reduction" if price_change < 0 else "increase"} '
        f"moves price closer to the revenue curve peak identified during model training."
    )
