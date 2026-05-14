"""
app.py — Streamlit dashboard for the Avocado Pricing Engine.

Run from the project root:
    streamlit run src/dashboard/app.py

Tabs:
    Forecast  — 12-week price forecast, 8 models + Ensemble_weighted conformal PI band
    Pricing   — revenue-optimal price recommendation and elasticity metrics
    Explain   — top-3 SHAP demand drivers for the selected series
    Simulate  — live LightGBM inference: adjust price, see predicted volume
"""
from __future__ import annotations

import math
import pathlib
import sys

# Streamlit adds src/dashboard/ to sys.path[0] when launching this file directly.
# Insert the project root so all `from src.*` imports resolve correctly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.charts import forecast_chart, shap_chart
from src.data.loader import load_datastore

st.set_page_config(
    page_title="Avocado Pricing Engine",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading models and data...")
def get_datastore():
    return load_datastore()


ds = get_datastore()

# ---------------------------------------------------------------------------
# Sidebar — series selector
# ---------------------------------------------------------------------------
st.sidebar.title("Avocado Pricing Engine")
st.sidebar.markdown("Select a market series to explore.")

series_options = sorted(ds.series_meta.keys())
default_idx = series_options.index("Albany_conventional")
selected_uid = st.sidebar.selectbox("Market series", series_options, index=default_idx)
meta = ds.series_meta[selected_uid]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Region:** {meta['region']}")
st.sidebar.markdown(f"**Type:** {meta['avocado_type'].capitalize()}")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title(f"{meta['region']} — {meta['avocado_type'].capitalize()}")
st.caption(
    "Dynamic pricing recommendations powered by LightGBM demand modeling "
    "and conformal prediction intervals."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_pricing, tab_explain, tab_simulate = st.tabs(
    ["Forecast", "Pricing", "Explain", "Simulate"]
)

# ── Forecast ─────────────────────────────────────────────────────────────────
with tab_forecast:
    series_df = ds.forecast_df[ds.forecast_df["unique_id"] == selected_uid].copy()
    pi_row = ds.pi_df[ds.pi_df["unique_id"] == selected_uid].iloc[0]

    st.plotly_chart(
        forecast_chart(series_df, float(pi_row["lower_q"]), float(pi_row["upper_q"])),
        use_container_width=True,
    )

    with st.expander("Raw forecast table"):
        model_cols = [c for c in ["Ensemble_weighted", "MSTL_ETS", "MSTL_ARIMA",
                                   "MSTL_Theta", "NHITS", "NBEATSx", "DLinear",
                                   "SeasonalNaive"] if c in series_df.columns]
        display = series_df[["ds"] + model_cols].copy()
        display.columns = ["Week"] + model_cols
        st.dataframe(
            display.set_index("Week").style.format("{:.3f}"),
            use_container_width=True,
        )

# ── Pricing ───────────────────────────────────────────────────────────────────
with tab_pricing:
    rec = ds.recommendations_df[
        ds.recommendations_df["unique_id"] == selected_uid
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${rec['current_price']:.2f}")
    c2.metric(
        "Optimal Price",
        f"${rec['optimal_price']:.2f}",
        delta=f"{rec['price_change_pct']:+.1f}%",
    )
    c3.metric("Price Elasticity", f"{rec['elasticity']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Current Revenue", f"${rec['current_revenue']:,.0f}")
    c5.metric(
        "Optimal Revenue",
        f"${rec['optimal_revenue']:,.0f}",
        delta=f"{rec['revenue_change_pct']:+.1f}%",
    )
    c6.metric("Type", "Organic" if rec["is_organic"] else "Conventional")

    st.markdown("---")
    st.markdown(
        "The optimal price maximises expected revenue within a ±30% search window "
        "around the current price, using the LightGBM demand model trained in "
        "notebook 04. Elasticity < −1 indicates elastic demand; "
        "−1 < e < 0 indicates inelastic demand."
    )

# ── Explain ───────────────────────────────────────────────────────────────────
with tab_explain:
    drivers = (
        ds.shap_drivers_df[ds.shap_drivers_df["unique_id"] == selected_uid]
        .sort_values("driver_rank")
    )

    st.plotly_chart(shap_chart(drivers), use_container_width=True)

    for _, row in drivers.iterrows():
        sign = "+" if row["direction"] == "increases_demand" else "-"
        st.markdown(
            f"**#{int(row['driver_rank'])} {row['feature']}** "
            f"= `{row['feature_value']:.3f}` — "
            f"SHAP {row['shap_value']:+.4f} ({sign} demand)"
        )

# ── Simulate ──────────────────────────────────────────────────────────────────
with tab_simulate:
    current_price = float(
        ds.recommendations_df[
            ds.recommendations_df["unique_id"] == selected_uid
        ].iloc[0]["current_price"]
    )

    test_price = st.slider(
        "Test price (USD per avocado)",
        min_value=round(current_price * 0.5, 2),
        max_value=round(current_price * 1.8, 2),
        value=current_price,
        step=0.05,
    )

    # Live LightGBM inference — swap AveragePrice in the latest feature row.
    ctx_row = ds.latest_ctx_df[
        ds.latest_ctx_df["unique_id"] == selected_uid
    ].iloc[0].copy()
    current_volume = float(ctx_row["Total Volume"])
    ctx_row["AveragePrice"] = test_price

    bundle = ds.model_bundle
    X = ctx_row[bundle["features"]].to_frame().T.astype(float)
    pred_log_vol = float(bundle["model"].predict(X)[0])
    pred_vol = math.exp(pred_log_vol)

    vol_delta_pct = (pred_vol - current_volume) / current_volume * 100
    rev_current = current_price * current_volume
    rev_simulated = test_price * pred_vol
    rev_delta_pct = (rev_simulated - rev_current) / rev_current * 100

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Test Price",
        f"${test_price:.2f}",
        delta=f"{(test_price - current_price) / current_price * 100:+.1f}% vs current",
    )
    m2.metric("Predicted Volume", f"{pred_vol:,.0f}", delta=f"{vol_delta_pct:+.1f}%")
    m3.metric("Simulated Revenue", f"${rev_simulated:,.0f}", delta=f"{rev_delta_pct:+.1f}%")

    st.caption(
        f"Baseline — price: ${current_price:.2f} | "
        f"volume: {current_volume:,.0f} | "
        f"revenue: ${rev_current:,.0f}"
    )
