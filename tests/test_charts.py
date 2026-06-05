"""
Unit tests for dashboard chart builders — no model artifacts required.

Each function in src/dashboard/charts.py is a pure transformation
(DataFrame(s) in → go.Figure out), so these tests always run in CI.

Coverage:
  - clean_forecast_chart: figure structure, trace types, axis labels, band polygon
  - revenue_comparison_chart: bar values, dollar annotation, delta=0 edge case
  - strategy_range_chart: four-category bar, y-values, axis label
  - drivers_plain_chart: horizontal orientation, color-by-direction, label lookup
"""

from __future__ import annotations

import datetime

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard.charts import (
    clean_forecast_chart,
    drivers_plain_chart,
    revenue_comparison_chart,
    strategy_range_chart,
)

# Color constants mirrored from charts.py.
# If these change in the source, the color tests will catch it.
_GREEN = "#43A047"
_RED = "#E53935"


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _forecast_df(n: int = 3) -> pd.DataFrame:
    """Minimal forecast DataFrame with required columns for clean_forecast_chart."""
    base = datetime.date(2018, 1, 7)
    return pd.DataFrame(
        {
            "ds": [base + datetime.timedelta(days=7 * i) for i in range(n)],
            "Ensemble_weighted": [1.40 + i * 0.02 for i in range(n)],
        }
    )


def _drivers_df() -> pd.DataFrame:
    """Three-row SHAP driver DataFrame for drivers_plain_chart."""
    return pd.DataFrame(
        {
            "feature": ["AveragePrice", "region_encoded", "lag_1"],
            "abs_shap_value": [0.35, 0.20, 0.10],
            "direction": ["decreases_demand", "increases_demand", "decreases_demand"],
        }
    )


# ---------------------------------------------------------------------------
# clean_forecast_chart
# ---------------------------------------------------------------------------


class TestCleanForecastChart:
    def test_returns_figure(self):
        fig = clean_forecast_chart(_forecast_df(), lower_q=-0.20, upper_q=0.15)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self):
        fig = clean_forecast_chart(_forecast_df(), lower_q=-0.20, upper_q=0.15)
        assert len(fig.data) == 2

    def test_first_trace_is_uncertainty_band(self):
        band = clean_forecast_chart(_forecast_df(), lower_q=-0.20, upper_q=0.15).data[0]
        assert isinstance(band, go.Scatter)
        assert band.fill == "toself"
        assert band.name == "Expected range"

    def test_second_trace_is_forecast_line(self):
        line = clean_forecast_chart(_forecast_df(), lower_q=-0.20, upper_q=0.15).data[1]
        assert isinstance(line, go.Scatter)
        assert line.name == "Expected price"

    def test_forecast_y_values_match_input(self):
        df = _forecast_df(n=3)
        line = clean_forecast_chart(df, lower_q=-0.20, upper_q=0.15).data[1]
        assert list(line.y) == pytest.approx([1.40, 1.42, 1.44])

    def test_band_x_length_is_twice_series_length(self):
        """Uncertainty band is a closed polygon: forward dates + reversed dates."""
        df = _forecast_df(n=4)
        band = clean_forecast_chart(df, lower_q=-0.20, upper_q=0.15).data[0]
        assert len(band.x) == 2 * len(df)

    def test_x_axis_title(self):
        fig = clean_forecast_chart(_forecast_df(), lower_q=-0.20, upper_q=0.15)
        assert fig.layout.xaxis.title.text == "Week"

    def test_y_axis_title(self):
        fig = clean_forecast_chart(_forecast_df(), lower_q=-0.20, upper_q=0.15)
        assert fig.layout.yaxis.title.text == "Price (USD per avocado)"

    def test_negative_lower_q_shifts_band_down(self):
        """Lower band = Ensemble_weighted + lower_q; must be below the line."""
        df = _forecast_df(n=1)
        lower_q, upper_q = -0.20, 0.15
        fig = clean_forecast_chart(df, lower_q=lower_q, upper_q=upper_q)
        ensemble = df["Ensemble_weighted"].iloc[0]
        # Band polygon contains both upper and lower values; lower must be < line.
        band_y = list(fig.data[0].y)
        assert min(band_y) == pytest.approx(ensemble + lower_q)
        assert max(band_y) == pytest.approx(ensemble + upper_q)


# ---------------------------------------------------------------------------
# revenue_comparison_chart
# ---------------------------------------------------------------------------


class TestRevenueComparisonChart:
    def test_returns_figure(self):
        assert isinstance(revenue_comparison_chart(100_000, 120_000, 1.50, 1.70), go.Figure)

    def test_has_one_trace(self):
        fig = revenue_comparison_chart(100_000, 120_000, 1.50, 1.70)
        assert len(fig.data) == 1

    def test_trace_is_bar(self):
        fig = revenue_comparison_chart(100_000, 120_000, 1.50, 1.70)
        assert isinstance(fig.data[0], go.Bar)

    def test_bar_y_values_match_inputs(self):
        fig = revenue_comparison_chart(100_000, 120_000, 1.50, 1.70)
        assert list(fig.data[0].y) == pytest.approx([100_000, 120_000])

    def test_bar_text_contains_dollar_sign(self):
        fig = revenue_comparison_chart(100_000, 120_000, 1.50, 1.70)
        assert all("$" in t for t in fig.data[0].text)

    def test_annotation_present_when_delta_positive(self):
        fig = revenue_comparison_chart(100_000, 120_000, 1.50, 1.70)
        assert len(fig.layout.annotations) == 1
        assert "$+20,000/week" == fig.layout.annotations[0].text

    def test_annotation_present_when_delta_negative(self):
        fig = revenue_comparison_chart(120_000, 100_000, 1.70, 1.50)
        assert len(fig.layout.annotations) == 1
        assert "$-20,000/week" == fig.layout.annotations[0].text

    def test_no_annotation_when_revenues_equal(self):
        """delta == 0 → no annotation added."""
        fig = revenue_comparison_chart(100_000, 100_000, 1.50, 1.50)
        assert len(fig.layout.annotations) == 0

    def test_x_labels_contain_prices(self):
        fig = revenue_comparison_chart(100_000, 120_000, 1.50, 1.70)
        x_labels = list(fig.data[0].x)
        assert "$1.50" in x_labels[0]
        assert "$1.70" in x_labels[1]


# ---------------------------------------------------------------------------
# strategy_range_chart
# ---------------------------------------------------------------------------


class TestStrategyRangeChart:
    def test_returns_figure(self):
        assert isinstance(strategy_range_chart(90_000, 120_000, 150_000, 110_000), go.Figure)

    def test_has_one_trace(self):
        fig = strategy_range_chart(90_000, 120_000, 150_000, 110_000)
        assert len(fig.data) == 1

    def test_trace_is_bar(self):
        fig = strategy_range_chart(90_000, 120_000, 150_000, 110_000)
        assert isinstance(fig.data[0], go.Bar)

    def test_has_four_categories(self):
        fig = strategy_range_chart(90_000, 120_000, 150_000, 110_000)
        assert len(fig.data[0].y) == 4

    def test_y_values_match_inputs_in_order(self):
        p10, p50, p90, cur = 90_000, 120_000, 150_000, 110_000
        fig = strategy_range_chart(p10, p50, p90, cur)
        assert list(fig.data[0].y) == pytest.approx([p10, p50, p90, cur])

    def test_y_axis_title(self):
        fig = strategy_range_chart(90_000, 120_000, 150_000, 110_000)
        assert fig.layout.yaxis.title.text == "Weekly Revenue (USD)"

    def test_bar_text_contains_dollar_signs(self):
        fig = strategy_range_chart(90_000, 120_000, 150_000, 110_000)
        assert all("$" in t for t in fig.data[0].text)


# ---------------------------------------------------------------------------
# drivers_plain_chart
# ---------------------------------------------------------------------------


class TestDriversPlainChart:
    def test_returns_figure(self):
        assert isinstance(drivers_plain_chart(_drivers_df(), {}), go.Figure)

    def test_has_one_trace(self):
        fig = drivers_plain_chart(_drivers_df(), {})
        assert len(fig.data) == 1

    def test_trace_is_bar(self):
        assert isinstance(drivers_plain_chart(_drivers_df(), {}).data[0], go.Bar)

    def test_trace_is_horizontal(self):
        fig = drivers_plain_chart(_drivers_df(), {})
        assert fig.data[0].orientation == "h"

    def test_decreases_demand_colored_red(self):
        df = pd.DataFrame(
            {
                "feature": ["AveragePrice"],
                "abs_shap_value": [0.35],
                "direction": ["decreases_demand"],
            }
        )
        fig = drivers_plain_chart(df, {})
        assert fig.data[0].marker.color[0] == _RED

    def test_increases_demand_colored_green(self):
        df = pd.DataFrame(
            {
                "feature": ["region_encoded"],
                "abs_shap_value": [0.20],
                "direction": ["increases_demand"],
            }
        )
        fig = drivers_plain_chart(df, {})
        assert fig.data[0].marker.color[0] == _GREEN

    def test_mixed_directions_get_correct_colors(self):
        df = _drivers_df()  # ["decreases", "increases", "decreases"]
        colors = list(drivers_plain_chart(df, {}).data[0].marker.color)
        assert colors == [_RED, _GREEN, _RED]

    def test_feature_label_lookup_applied(self):
        df = pd.DataFrame(
            {
                "feature": ["AveragePrice"],
                "abs_shap_value": [0.35],
                "direction": ["decreases_demand"],
            }
        )
        fig = drivers_plain_chart(df, {"AveragePrice": "Selling price"})
        assert list(fig.data[0].y) == ["Selling price"]

    def test_unknown_feature_falls_back_to_title_case(self):
        df = pd.DataFrame(
            {
                "feature": ["some_unknown_feature"],
                "abs_shap_value": [0.10],
                "direction": ["increases_demand"],
            }
        )
        fig = drivers_plain_chart(df, {})
        assert list(fig.data[0].y) == ["Some Unknown Feature"]

    def test_x_values_are_abs_shap(self):
        df = _drivers_df()
        fig = drivers_plain_chart(df, {})
        assert list(fig.data[0].x) == pytest.approx([0.35, 0.20, 0.10])

    def test_partial_label_dict_falls_back_for_missing(self):
        """Labels dict covers some features; unknown ones should fall back."""
        df = pd.DataFrame(
            {
                "feature": ["AveragePrice", "some_feature"],
                "abs_shap_value": [0.35, 0.10],
                "direction": ["decreases_demand", "increases_demand"],
            }
        )
        fig = drivers_plain_chart(df, {"AveragePrice": "Selling price"})
        labels = list(fig.data[0].y)
        assert labels[0] == "Selling price"
        assert labels[1] == "Some Feature"
