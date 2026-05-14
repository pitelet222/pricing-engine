"""
charts.py — Reusable Plotly figure builders for the Streamlit dashboard.

Each function is a pure transformation: DataFrame(s) in, go.Figure out.
No Streamlit calls here — keeps chart logic testable and composable.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_MODEL_COLORS = {
    "Ensemble_weighted": "#2196F3",
    "MSTL_ETS":          "#4CAF50",
    "MSTL_ARIMA":        "#FF9800",
    "MSTL_Theta":        "#9C27B0",
    "NHITS":             "#F44336",
    "NBEATSx":           "#00BCD4",
    "DLinear":           "#FF5722",
    "SeasonalNaive":     "#9E9E9E",
}
_PI_FILL = "rgba(33, 150, 243, 0.15)"


def forecast_chart(
    series_df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> go.Figure:
    """
    Multi-line price forecast with a shaded Ensemble_weighted conformal PI band.

    Parameters
    ----------
    series_df : DataFrame with columns ds, Ensemble_weighted, MSTL_ETS, MSTL_ARIMA, ...
    lower_q   : additive lower-bound offset (from conformal_pi_stats.csv)
    upper_q   : additive upper-bound offset
    """
    fig = go.Figure()

    # Shaded PI band — drawn first so model lines render on top.
    upper_vals = series_df["Ensemble_weighted"] + upper_q
    lower_vals = series_df["Ensemble_weighted"] + lower_q
    dates = list(series_df["ds"])
    fig.add_trace(go.Scatter(
        x=dates + list(reversed(dates)),
        y=list(upper_vals) + list(reversed(list(lower_vals))),
        fill="toself",
        fillcolor=_PI_FILL,
        line=dict(color="rgba(0,0,0,0)"),
        name="Ensemble 80% PI",
        hoverinfo="skip",
    ))

    for model, color in _MODEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=series_df["ds"],
            y=series_df[model],
            name=model,
            line=dict(color=color, width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ))

    fig.update_layout(
        title="12-Week Price Forecast",
        xaxis_title="Week",
        yaxis_title="Avg Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=420,
        margin=dict(t=80),
    )
    return fig


def shap_chart(drivers_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of top-3 SHAP demand drivers.

    Green bars increase demand; red bars decrease it.
    Feature value is appended to the y-axis label for context.
    """
    colors = [
        "#4CAF50" if d == "increases_demand" else "#F44336"
        for d in drivers_df["direction"]
    ]
    labels = [
        f"{row['feature']}  (= {row['feature_value']:.3f})"
        for _, row in drivers_df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=drivers_df["shap_value"],
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Top-3 Demand Drivers (SHAP Values)",
        xaxis_title="SHAP value — impact on log(volume)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=220),
    )
    return fig
