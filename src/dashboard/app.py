"""
app.py — Business-focused Streamlit dashboard for the Avocado Pricing Engine.

Designed for a non-technical audience: one clear recommendation, revenue impact
in dollars, interactive price simulator, and plain-language demand insights.

Run from the project root:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import math
import pathlib
import sys
from collections import defaultdict

# Streamlit adds src/dashboard/ to sys.path[0]; insert project root for src.* imports.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.charts import (
    clean_forecast_chart,
    drivers_plain_chart,
    revenue_comparison_chart,
    strategy_range_chart,
)
from src.data.loader import load_datastore

st.set_page_config(
    page_title="Avocado Pricing Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading models and data...")
def get_datastore():
    return load_datastore()


ds = get_datastore()

# ---------------------------------------------------------------------------
# Build region / type lookup from series_meta
# ---------------------------------------------------------------------------
region_to_types: dict[str, list[str]] = defaultdict(list)
for _uid, _meta in ds.series_meta.items():
    region_to_types[_meta["region"]].append(_meta["avocado_type"])

sorted_regions = sorted(region_to_types.keys())

# ---------------------------------------------------------------------------
# Sidebar — market selector and KPI summary
# ---------------------------------------------------------------------------
st.sidebar.title("Avocado Pricing Engine")
st.sidebar.caption("Revenue optimisation powered by AI forecasting")
st.sidebar.markdown("---")

selected_region = st.sidebar.selectbox(
    "Market region",
    sorted_regions,
    index=sorted_regions.index("Albany") if "Albany" in sorted_regions else 0,
)

available_types = sorted(region_to_types[selected_region])
_TYPE_LABEL = {"conventional": "Conventional", "organic": "Organic"}
selected_type_label = st.sidebar.radio(
    "Avocado type",
    [_TYPE_LABEL[t] for t in available_types],
)
selected_type = next(k for k, v in _TYPE_LABEL.items() if v == selected_type_label)
selected_uid = f"{selected_region}_{selected_type}"

# Fetch pre-computed rows for the selected series
rec = ds.recommendations_df[ds.recommendations_df["unique_id"] == selected_uid].iloc[0]
unc = ds.uncertainty_df[ds.uncertainty_df["unique_id"] == selected_uid].iloc[0]

st.sidebar.markdown("---")
st.sidebar.metric("Current price", f"${float(rec['current_price']):.2f}")
st.sidebar.metric(
    "Recommended price",
    f"${float(rec['optimal_price']):.2f}",
    delta=f"{float(rec['price_change_pct']):+.1f}%",
)
_sidebar_rev_gain = float(rec["optimal_revenue"]) - float(rec["current_revenue"])
st.sidebar.metric(
    "Weekly revenue opportunity",
    f"${_sidebar_rev_gain:+,.0f}",
    delta=f"{float(rec['revenue_change_pct']):+.1f}%",
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title(f"{selected_region} — {selected_type_label} Avocados")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rec, tab_sim, tab_risk, tab_why = st.tabs([
    "Pricing Recommendation",
    "Price Simulator",
    "Risk & Strategies",
    "What Drives Demand?",
])

# ── Pricing Recommendation ────────────────────────────────────────────────────
with tab_rec:
    price_delta_pct = float(rec["price_change_pct"])
    rev_gain = float(rec["optimal_revenue"]) - float(rec["current_revenue"])

    if price_delta_pct > 0:
        action_verb, banner = "Raise", st.success
    elif price_delta_pct < 0:
        action_verb, banner = "Lower", st.warning
    else:
        action_verb, banner = "Maintain", st.info

    banner(
        f"**{action_verb} your price from ${float(rec['current_price']):.2f} "
        f"to ${float(rec['optimal_price']):.2f}** ({price_delta_pct:+.1f}%) — "
        f"expected to generate **${rev_gain:+,.0f} more per week** "
        f"({float(rec['revenue_change_pct']):+.1f}%)."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current weekly revenue", f"${float(rec['current_revenue']):,.0f}")
    col2.metric(
        "Revenue at recommended price",
        f"${float(rec['optimal_revenue']):,.0f}",
        delta=f"${rev_gain:+,.0f} ({float(rec['revenue_change_pct']):+.1f}%)",
    )

    # Elasticity translated to plain language — the number itself stays hidden.
    e = float(rec["elasticity"])
    if e < -2.0:
        loyalty_label = "Very price-sensitive"
        loyalty_detail = (
            "Customers in this market react strongly to price changes. "
            "Even small increases can noticeably reduce sales."
        )
    elif e < -1.0:
        loyalty_label = "Price-sensitive"
        loyalty_detail = (
            "Customers are moderately sensitive to price. "
            "The model balances higher margins against the risk of lower volume."
        )
    elif e < -0.5:
        loyalty_label = "Moderately loyal"
        loyalty_detail = (
            "Most customers will absorb small price increases "
            "without reducing purchases significantly."
        )
    else:
        loyalty_label = "Loyal customer base"
        loyalty_detail = (
            "Customers in this market are relatively insensitive to price changes — "
            "a good position for margin improvement."
        )

    col3.metric("Customer price sensitivity", loyalty_label)
    st.caption(f"*{loyalty_detail}*")

    st.markdown("---")

    col_chart, _ = st.columns([3, 1])
    with col_chart:
        st.subheader("Current vs. recommended revenue")
        st.plotly_chart(
            revenue_comparison_chart(
                float(rec["current_revenue"]),
                float(rec["optimal_revenue"]),
                float(rec["current_price"]),
                float(rec["optimal_price"]),
            ),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Expected price trend — next 12 weeks")
    st.caption(
        "Forecast from the ensemble model. "
        "The shaded area shows the expected price range — "
        "roughly 80% of outcomes should fall within this band."
    )
    series_df = ds.forecast_df[ds.forecast_df["unique_id"] == selected_uid].copy()
    pi_row = ds.pi_df[ds.pi_df["unique_id"] == selected_uid].iloc[0]
    st.plotly_chart(
        clean_forecast_chart(
            series_df,
            float(pi_row["lower_q"]),
            float(pi_row["upper_q"]),
        ),
        use_container_width=True,
    )

# ── Price Simulator ────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("What if I change my price?")
    st.markdown(
        "Drag the slider to test any price point. "
        "The AI model instantly estimates how your sales volume and weekly revenue would change."
    )

    current_price = float(rec["current_price"])
    optimal_price = float(rec["optimal_price"])

    test_price = st.slider(
        "Test price (USD per avocado)",
        min_value=round(current_price * 0.5, 2),
        max_value=round(current_price * 1.8, 2),
        value=round(optimal_price, 2),  # default to the recommendation
        step=0.05,
        help="Move the slider to see how revenue changes at different price points.",
    )

    # Live LightGBM inference — swap AveragePrice, keep everything else fixed.
    ctx_row = ds.latest_ctx_df[
        ds.latest_ctx_df["unique_id"] == selected_uid
    ].iloc[0].copy()
    current_volume = float(ctx_row["Total Volume"])
    ctx_row["AveragePrice"] = test_price

    bundle = ds.model_bundle
    X = ctx_row[bundle["features"]].to_frame().T.astype(float)
    pred_log_vol = float(bundle["model"].predict(X)[0])
    pred_vol = math.exp(pred_log_vol)

    rev_current = current_price * current_volume
    rev_simulated = test_price * pred_vol
    rev_delta = rev_simulated - rev_current
    rev_delta_pct = rev_delta / rev_current * 100
    vol_delta_pct = (pred_vol - current_volume) / current_volume * 100

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Test price",
        f"${test_price:.2f}",
        delta=f"{(test_price - current_price) / current_price * 100:+.1f}% vs current",
    )
    col2.metric(
        "Expected weekly revenue",
        f"${rev_simulated:,.0f}",
        delta=f"${rev_delta:+,.0f} ({rev_delta_pct:+.1f}%)",
    )
    col3.metric(
        "Expected weekly sales volume",
        f"{pred_vol:,.0f} units",
        delta=f"{vol_delta_pct:+.1f}%",
    )

    st.markdown("---")
    col_cur, col_sim = st.columns(2)
    with col_cur:
        st.markdown("**Current setup**")
        st.markdown(f"- Price: **${current_price:.2f}** per unit")
        st.markdown(f"- Volume: **{current_volume:,.0f}** units/week")
        st.markdown(f"- Revenue: **${rev_current:,.0f}**/week")
    with col_sim:
        st.markdown("**Simulated setup**")
        st.markdown(f"- Price: **${test_price:.2f}** per unit")
        st.markdown(f"- Volume: **{pred_vol:,.0f}** units/week")
        st.markdown(f"- Revenue: **${rev_simulated:,.0f}**/week")

    # Nudge toward the recommendation if the user has moved away from it
    if abs(test_price - optimal_price) > 0.02:
        st.info(
            f"The model recommends **${optimal_price:.2f}** as the optimal price for this market. "
            f"Try sliding to that point to see the full revenue opportunity."
        )

# ── Risk & Strategies ─────────────────────────────────────────────────────────
with tab_risk:
    st.subheader("Three strategies — choose your level of ambition")
    st.markdown(
        "The model ran thousands of price simulations across different demand scenarios. "
        "Below are three pricing strategies, from most cautious to most aggressive."
    )

    strategies_agree = bool(unc.get("strategies_agree", False))
    if strategies_agree:
        st.success(
            "All three strategies agree on the price direction. "
            "This is a high-confidence recommendation — the signal is robust."
        )
    else:
        st.warning(
            "The three strategies suggest different price directions. "
            "There is more uncertainty in this market — the conservative option is safer."
        )

    st.markdown("---")

    _STRATEGIES = [
        (
            "conservative",
            "Conservative",
            "Protects current revenue. Smaller price move, lower risk.",
        ),
        (
            "balanced",
            "Balanced (Recommended)",
            "Maximises expected revenue. The default recommendation.",
        ),
        (
            "aggressive",
            "Aggressive",
            "Goes for maximum upside. Higher potential reward, but also higher risk.",
        ),
    ]

    cols = st.columns(3)
    for col, (key, label, description) in zip(cols, _STRATEGIES):
        opt_price = float(unc[f"opt_price_{key}"])
        uplift = float(unc[f"rev_uplift_{key}"])

        # price_direction exists for conservative and aggressive but not balanced
        dir_col = f"price_direction_{key}"
        if dir_col in unc.index:
            direction = int(unc[dir_col])
        else:
            direction = 1 if opt_price > float(unc["current_price"]) else -1
        price_dir_text = "Raise price" if direction == 1 else "Lower price"

        with col:
            st.markdown(f"**{label}**")
            st.caption(description)
            st.metric("Recommended price", f"${opt_price:.2f}")
            st.metric("Expected revenue uplift", f"{uplift:+.1f}%")
            st.markdown(f"Direction: **{price_dir_text}**")

    st.markdown("---")
    st.subheader("Revenue range across all simulated scenarios")
    st.caption(
        "Shows how wide the revenue uncertainty is at the current price. "
        "A narrow gap between pessimistic and optimistic means the market is predictable."
    )

    rev_p10 = float(unc["rev_p10_current"])
    rev_p50 = float(unc["rev_p50_current"])
    rev_p90 = float(unc["rev_p90_current"])
    current_rev = float(rec["current_revenue"])

    col_chart2, col_stats = st.columns([2, 1])
    with col_chart2:
        st.plotly_chart(
            strategy_range_chart(rev_p10, rev_p50, rev_p90, current_rev),
            use_container_width=True,
        )
    with col_stats:
        st.metric("Pessimistic scenario (10th pct)", f"${rev_p10:,.0f}")
        st.metric("Most likely outcome (median)", f"${rev_p50:,.0f}")
        st.metric("Optimistic scenario (90th pct)", f"${rev_p90:,.0f}")
        downside = float(unc["downside_risk_pct"])
        st.metric(
            "Downside risk",
            f"{downside:.1f}%",
            help="Probability that revenue falls below the current level under uncertainty.",
        )

# ── What Drives Demand? ────────────────────────────────────────────────────────
with tab_why:
    st.subheader("What most influences sales in this market?")
    st.markdown(
        "These are the three factors that have the biggest impact on how many "
        "avocados are sold each week. Understanding them helps anticipate "
        "demand changes before they happen."
    )

    drivers = (
        ds.shap_drivers_df[ds.shap_drivers_df["unique_id"] == selected_uid]
        .sort_values("driver_rank")
    )

    # Technical feature names → plain business language
    _FEATURE_LABELS: dict[str, str] = {
        "AveragePrice": "Selling price",
        "Total Volume": "Total sales volume",
        "4046": "Small avocado sales",
        "4225": "Large avocado sales",
        "4770": "Extra-large avocado sales",
        "Total Bags": "Total bags sold",
        "Small Bags": "Small bag sales",
        "Large Bags": "Large bag sales",
        "XLarge Bags": "Extra-large bag sales",
        "year": "Year",
        "month": "Month of year",
        "week_of_year": "Time of year (seasonality)",
        "is_organic": "Organic product",
        "region_encoded": "Regional market",
        "lag_1": "Price from last week",
        "lag_4": "Price from 4 weeks ago",
        "lag_12": "Price from 12 weeks ago",
        "rolling_mean_4": "Average price — last 4 weeks",
        "rolling_mean_12": "Average price — last 12 weeks",
        "rolling_std_4": "Price stability (last 4 weeks)",
        "price_trend": "Recent price trend",
    }

    st.plotly_chart(
        drivers_plain_chart(drivers, _FEATURE_LABELS),
        use_container_width=True,
    )

    st.markdown("---")
    for _, row in drivers.iterrows():
        label = _FEATURE_LABELS.get(
            row["feature"], row["feature"].replace("_", " ").title()
        )
        is_positive = row["direction"] == "increases_demand"
        direction_text = "boosts" if is_positive else "reduces"
        value = float(row["feature_value"])
        feat = row["feature"]

        # Format value in a human-friendly way based on the feature type
        if feat == "AveragePrice" or "price" in feat.lower() or "lag" in feat or "rolling" in feat:
            value_str = f"${value:.2f}"
        elif feat in ("year",):
            value_str = str(int(value))
        elif feat in ("month", "week_of_year"):
            value_str = str(int(value))
        elif feat == "is_organic":
            value_str = "Yes" if value == 1 else "No"
        elif any(k in feat for k in ("Volume", "Bags", "4046", "4225", "4770")):
            value_str = f"{value:,.0f}"
        else:
            value_str = f"{value:.2f}"

        color = "green" if is_positive else "red"
        st.markdown(
            f"**#{int(row['driver_rank'])} — {label}** "
            f"(current value: {value_str}) "
            f"— :{color}[{direction_text.capitalize()} weekly sales in this market]"
        )
