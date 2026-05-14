"""
charts.py — Reusable Plotly figure builders for the Streamlit dashboard.

Each function is a pure transformation: DataFrame(s) in, go.Figure out.
No Streamlit calls here — keeps chart logic testable and composable.

Designed for a non-technical audience: plain labels, revenue in dollars,
one clear signal per chart, no model-comparison clutter.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_BLUE = "#2196F3"
_BLUE_LIGHT = "rgba(33, 150, 243, 0.12)"
_BLUE_MID = "rgba(33, 150, 243, 0.45)"
_GREEN = "#43A047"
_GREEN_LIGHT = "#A5D6A7"
_RED = "#E53935"
_RED_LIGHT = "#EF9A9A"
_GREY = "#9E9E9E"
_GREY_LIGHT = "#BDBDBD"
_BG = "white"
_GRID = "#EEEEEE"


def clean_forecast_chart(
    series_df: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> go.Figure:
    """
    Single-line price forecast (ensemble only) with a shaded uncertainty band.

    Hides all individual model lines — a business audience cares about
    "where is the price going?", not which model predicts what.
    """
    fig = go.Figure()

    upper_vals = series_df["Ensemble_weighted"] + upper_q
    lower_vals = series_df["Ensemble_weighted"] + lower_q
    dates = list(series_df["ds"])

    # Uncertainty band — drawn first so the line renders on top
    fig.add_trace(go.Scatter(
        x=dates + list(reversed(dates)),
        y=list(upper_vals) + list(reversed(list(lower_vals))),
        fill="toself",
        fillcolor=_BLUE_LIGHT,
        line=dict(color="rgba(0,0,0,0)"),
        name="Expected range",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=series_df["ds"],
        y=series_df["Ensemble_weighted"],
        name="Expected price",
        line=dict(color=_BLUE, width=3),
        mode="lines+markers",
        marker=dict(size=7, color=_BLUE),
        hovertemplate="Week: %{x|%b %d, %Y}<br>Expected price: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Price (USD per avocado)",
        yaxis_tickprefix="$",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=360,
        margin=dict(t=30, b=40, l=60, r=20),
        plot_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
    )
    return fig


def revenue_comparison_chart(
    current_rev: float,
    optimal_rev: float,
    current_price: float,
    optimal_price: float,
) -> go.Figure:
    """
    Side-by-side bar comparing current vs. recommended weekly revenue.
    """
    labels = [
        f"Current price<br>${current_price:.2f}/unit",
        f"Recommended price<br>${optimal_price:.2f}/unit",
    ]
    values = [current_rev, optimal_rev]
    colors = ["#90CAF9", _GREEN]
    text_colors = ["#1565C0", "#1B5E20"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=15, color=text_colors),
        hovertemplate="%{x}<br>Revenue: $%{y:,.0f}<extra></extra>",
        width=0.45,
    ))

    # Add a subtle annotation for the uplift
    delta = optimal_rev - current_rev
    if delta != 0:
        fig.add_annotation(
            x=1,
            y=max(values) * 1.12,
            text=f"${delta:+,.0f}/week",
            showarrow=False,
            font=dict(size=14, color=_GREEN if delta > 0 else _RED),
        )

    max_y = max(values) * 1.25
    fig.update_layout(
        yaxis=dict(
            title="Weekly Revenue (USD)",
            tickprefix="$",
            gridcolor=_GRID,
            range=[0, max_y],
        ),
        height=320,
        margin=dict(t=50, b=40, l=80, r=20),
        showlegend=False,
        plot_bgcolor=_BG,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def strategy_range_chart(
    p10: float,
    p50: float,
    p90: float,
    current_rev: float,
) -> go.Figure:
    """
    Four-bar chart comparing pessimistic / likely / optimistic / current weekly revenue.

    Revenue percentiles represent the spread of outcomes at the current price
    under different demand scenarios derived from the conformal prediction intervals.
    """
    categories = [
        "Pessimistic<br>(10th pct)",
        "Most likely<br>(median)",
        "Optimistic<br>(90th pct)",
        "Current<br>revenue",
    ]
    values = [p10, p50, p90, current_rev]
    colors = [_RED_LIGHT, _BLUE, _GREEN_LIGHT, _GREY_LIGHT]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        width=0.5,
    ))

    fig.update_layout(
        yaxis=dict(
            title="Weekly Revenue (USD)",
            tickprefix="$",
            gridcolor=_GRID,
            range=[0, max(values) * 1.22],
        ),
        height=320,
        margin=dict(t=40, b=40, l=80, r=20),
        showlegend=False,
        plot_bgcolor=_BG,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def drivers_plain_chart(
    drivers_df: pd.DataFrame,
    feature_labels: dict[str, str],
) -> go.Figure:
    """
    Horizontal bar chart of top-3 demand drivers using plain-language labels.

    Bar length = relative importance (absolute SHAP).
    Green = boosts weekly sales; Red = reduces weekly sales.
    X-axis labels are hidden — the emphasis is on which factors matter most
    and in which direction, not on precise technical magnitudes.
    """
    colors = [
        _GREEN if d == "increases_demand" else _RED
        for d in drivers_df["direction"]
    ]
    plain_labels = [
        feature_labels.get(row["feature"], row["feature"].replace("_", " ").title())
        for _, row in drivers_df.iterrows()
    ]
    direction_texts = [
        "Boosts sales" if d == "increases_demand" else "Reduces sales"
        for d in drivers_df["direction"]
    ]

    fig = go.Figure(go.Bar(
        x=drivers_df["abs_shap_value"],
        y=plain_labels,
        orientation="h",
        marker_color=colors,
        hovertemplate=[
            f"<b>{label}</b><br>{dt}<br>Relative influence: {v:.3f}<extra></extra>"
            for label, dt, v in zip(
                plain_labels, direction_texts, drivers_df["abs_shap_value"]
            )
        ],
    ))

    fig.update_layout(
        xaxis=dict(
            title="Relative influence on weekly sales",
            showticklabels=False,
            gridcolor=_GRID,
        ),
        yaxis=dict(autorange="reversed"),
        height=260,
        margin=dict(l=240, r=20, t=30, b=40),
        showlegend=False,
        plot_bgcolor=_BG,
    )
    return fig
