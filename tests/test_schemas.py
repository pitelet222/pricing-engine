"""
Pure unit tests for Pydantic schemas — no model artifacts required.

These tests always run in CI. They verify that:
  - Schemas accept valid payloads without raising ValidationError
  - Field constraints are enforced (price > 0, driver_rank 1–3, etc.)
  - Enum literals reject unknown values
  - Optional fields accept None
"""

import datetime

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    ExplainResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    PricingStrategy,
    RecommendationResponse,
    SeriesItem,
    SeriesListResponse,
    ShapDriver,
    SimulateRequest,
    SimulateResponse,
    UncertaintyResponse,
)

# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_valid(self):
        h = HealthResponse(status="ok", series_loaded=86, version="0.1.0")
        assert h.status == "ok"
        assert h.series_loaded == 86

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="degraded", series_loaded=86, version="0.1.0")

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="ok", series_loaded=86)

    def test_manifest_ok_defaults_true(self):
        h = HealthResponse(status="ok", series_loaded=86, version="0.1.0")
        assert h.manifest_ok is True

    def test_manifest_ok_can_be_false(self):
        h = HealthResponse(status="ok", series_loaded=86, version="0.1.0", manifest_ok=False)
        assert h.manifest_ok is False

    def test_artifacts_generated_at_defaults_none(self):
        h = HealthResponse(status="ok", series_loaded=86, version="0.1.0")
        assert h.artifacts_generated_at is None

    def test_artifacts_generated_at_accepts_timestamp(self):
        h = HealthResponse(
            status="ok",
            series_loaded=86,
            version="0.1.0",
            artifacts_generated_at="2024-03-15T10:30:00+00:00",
        )
        assert h.artifacts_generated_at == "2024-03-15T10:30:00+00:00"


# ---------------------------------------------------------------------------
# SeriesItem / SeriesListResponse
# ---------------------------------------------------------------------------


class TestSeriesItem:
    def test_valid(self):
        s = SeriesItem(
            unique_id="Albany_conventional", region="Albany", avocado_type="conventional"
        )
        assert s.region == "Albany"

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            SeriesItem(unique_id="x", region="x", avocado_type="biodynamic")

    def test_both_valid_types(self):
        SeriesItem(unique_id="a", region="r", avocado_type="organic")
        SeriesItem(unique_id="b", region="r", avocado_type="conventional")

    def test_list_response(self):
        items = [
            SeriesItem(unique_id=f"s{i}", region="R", avocado_type="organic") for i in range(3)
        ]
        resp = SeriesListResponse(total=3, limit=3, offset=0, series=items)
        assert resp.total == 3
        assert resp.limit == 3
        assert resp.offset == 0
        assert len(resp.series) == 3

    def test_list_response_missing_limit_raises(self):
        items = [SeriesItem(unique_id="s0", region="R", avocado_type="organic")]
        with pytest.raises(ValidationError):
            SeriesListResponse(total=1, series=items)  # missing limit and offset


# ---------------------------------------------------------------------------
# ForecastPoint / ForecastResponse
# ---------------------------------------------------------------------------

_FORECAST_POINT_PAYLOAD = dict(
    ds=datetime.date(2018, 1, 7),
    ensemble_weighted=1.50,
    ensemble_lower=1.20,
    ensemble_upper=1.80,
    mstl_ets=1.48,
    mstl_arima=1.52,
    mstl_theta=1.49,
    nhits=1.51,
    nbeatsx=None,
    dlinear=None,
    seasonal_naive=1.45,
)


class TestForecastPoint:
    def test_valid_with_optional_none(self):
        pt = ForecastPoint(**_FORECAST_POINT_PAYLOAD)
        assert pt.nbeatsx is None
        assert pt.dlinear is None

    def test_valid_with_optional_set(self):
        payload = {**_FORECAST_POINT_PAYLOAD, "nbeatsx": 1.53, "dlinear": 1.47}
        pt = ForecastPoint(**payload)
        assert pt.nbeatsx == pytest.approx(1.53)
        assert pt.dlinear == pytest.approx(1.47)

    def test_missing_required_field(self):
        payload = {k: v for k, v in _FORECAST_POINT_PAYLOAD.items() if k != "nhits"}
        with pytest.raises(ValidationError):
            ForecastPoint(**payload)

    def test_date_field(self):
        pt = ForecastPoint(**_FORECAST_POINT_PAYLOAD)
        assert isinstance(pt.ds, datetime.date)


class TestForecastResponse:
    def test_default_pi_coverage(self):
        resp = ForecastResponse(unique_id="x", points=[])
        assert resp.pi_coverage == pytest.approx(0.90)

    def test_custom_pi_coverage(self):
        resp = ForecastResponse(unique_id="x", pi_coverage=0.80, points=[])
        assert resp.pi_coverage == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# RecommendationResponse
# ---------------------------------------------------------------------------


class TestRecommendationResponse:
    def test_valid(self):
        rec = RecommendationResponse(
            unique_id="Albany_conventional",
            is_organic=False,
            current_price=1.20,
            optimal_price=1.35,
            price_change_pct=12.5,
            current_revenue=50000.0,
            optimal_revenue=58000.0,
            revenue_change_pct=16.0,
            elasticity=-1.8,
        )
        assert rec.is_organic is False
        assert rec.elasticity == pytest.approx(-1.8)

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            RecommendationResponse(unique_id="x", is_organic=True)


# ---------------------------------------------------------------------------
# ShapDriver / ExplainResponse
# ---------------------------------------------------------------------------


class TestShapDriver:
    def test_valid(self):
        d = ShapDriver(
            driver_rank=1,
            feature="AveragePrice",
            shap_value=-0.35,
            abs_shap_value=0.35,
            direction="decreases_demand",
            feature_value=1.50,
        )
        assert d.driver_rank == 1

    def test_rank_below_1(self):
        with pytest.raises(ValidationError):
            ShapDriver(
                driver_rank=0,
                feature="x",
                shap_value=0.1,
                abs_shap_value=0.1,
                direction="increases_demand",
                feature_value=1.0,
            )

    def test_rank_above_3(self):
        with pytest.raises(ValidationError):
            ShapDriver(
                driver_rank=4,
                feature="x",
                shap_value=0.1,
                abs_shap_value=0.1,
                direction="increases_demand",
                feature_value=1.0,
            )

    def test_invalid_direction(self):
        with pytest.raises(ValidationError):
            ShapDriver(
                driver_rank=1,
                feature="x",
                shap_value=0.1,
                abs_shap_value=0.1,
                direction="neutral",
                feature_value=1.0,
            )

    def test_both_directions_valid(self):
        for direction in ("increases_demand", "decreases_demand"):
            ShapDriver(
                driver_rank=1,
                feature="x",
                shap_value=0.1,
                abs_shap_value=0.1,
                direction=direction,
                feature_value=1.0,
            )


class TestExplainResponse:
    def test_valid_three_drivers(self):
        drivers = [
            ShapDriver(
                driver_rank=i,
                feature=f"f{i}",
                shap_value=float(i) * 0.1,
                abs_shap_value=float(i) * 0.1,
                direction="increases_demand",
                feature_value=float(i),
            )
            for i in range(1, 4)
        ]
        resp = ExplainResponse(unique_id="x", drivers=drivers)
        assert len(resp.drivers) == 3


# ---------------------------------------------------------------------------
# SimulateRequest / SimulateResponse
# ---------------------------------------------------------------------------


class TestSimulateRequest:
    def test_valid(self):
        req = SimulateRequest(unique_id="Albany_conventional", price=1.50)
        assert req.price == pytest.approx(1.50)

    def test_zero_price_rejected(self):
        with pytest.raises(ValidationError):
            SimulateRequest(unique_id="x", price=0.0)

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            SimulateRequest(unique_id="x", price=-0.01)

    def test_very_small_positive_price_accepted(self):
        req = SimulateRequest(unique_id="x", price=0.001)
        assert req.price > 0


class TestSimulateResponse:
    def test_valid(self):
        resp = SimulateResponse(
            unique_id="Albany_conventional",
            price=1.50,
            predicted_log_volume=12.3,
            predicted_volume=219_895.0,
            current_price=1.20,
            current_volume=180_000.0,
        )
        assert resp.predicted_volume == pytest.approx(219_895.0)


# ---------------------------------------------------------------------------
# BatchRecommendRequest / BatchRecommendResponse
# ---------------------------------------------------------------------------


class TestBatchRecommendRequest:
    def test_valid(self):
        req = BatchRecommendRequest(unique_ids=["Albany_conventional", "LosAngeles_organic"])
        assert len(req.unique_ids) == 2

    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            BatchRecommendRequest(unique_ids=[])


_REC = RecommendationResponse(
    unique_id="Albany_conventional",
    is_organic=False,
    current_price=1.20,
    optimal_price=1.35,
    price_change_pct=12.5,
    current_revenue=50_000.0,
    optimal_revenue=58_000.0,
    revenue_change_pct=16.0,
    elasticity=-1.8,
)


class TestBatchRecommendResponse:
    def test_valid(self):
        resp = BatchRecommendResponse(
            requested=2,
            found=1,
            not_found=["UNKNOWN"],
            results=[_REC],
        )
        assert resp.requested == 2
        assert resp.found == 1
        assert resp.not_found == ["UNKNOWN"]
        assert len(resp.results) == 1

    def test_empty_results(self):
        resp = BatchRecommendResponse(requested=1, found=0, not_found=["X"], results=[])
        assert resp.found == 0
        assert resp.results == []


# ---------------------------------------------------------------------------
# PricingStrategy / UncertaintyResponse
# ---------------------------------------------------------------------------

_STRATEGY = PricingStrategy(opt_price=1.72, rev_uplift_pct=8.6)

_UNCERTAINTY_PAYLOAD = dict(
    unique_id="Albany_conventional",
    is_organic=False,
    current_price=1.57,
    conservative=_STRATEGY,
    balanced=_STRATEGY,
    aggressive=_STRATEGY,
    rev_p10=90_000.0,
    rev_p50=136_376.88,
    rev_p90=190_000.0,
    rev_spread_pct=73.5,
    downside_risk_pct=8.3,
    uplift_mean=8.2,
    uplift_std=3.1,
    uplift_sharpe=2.6,
    strategies_agree=True,
    price_direction_conservative=1,
    price_direction_aggressive=1,
)


class TestPricingStrategy:
    def test_valid(self):
        s = PricingStrategy(opt_price=1.80, rev_uplift_pct=12.1)
        assert s.opt_price == pytest.approx(1.80)

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            PricingStrategy(opt_price=1.80)


class TestUncertaintyResponse:
    def test_valid(self):
        resp = UncertaintyResponse(**_UNCERTAINTY_PAYLOAD)
        assert resp.unique_id == "Albany_conventional"
        assert resp.strategies_agree is True
        assert resp.price_direction_conservative == 1
        assert resp.price_direction_aggressive == 1

    def test_is_organic_bool(self):
        resp = UncertaintyResponse(**_UNCERTAINTY_PAYLOAD)
        assert isinstance(resp.is_organic, bool)
        assert resp.is_organic is False

    def test_nested_strategies(self):
        resp = UncertaintyResponse(**_UNCERTAINTY_PAYLOAD)
        assert isinstance(resp.conservative, PricingStrategy)
        assert isinstance(resp.balanced, PricingStrategy)
        assert isinstance(resp.aggressive, PricingStrategy)

    def test_missing_strategy_raises(self):
        payload = {k: v for k, v in _UNCERTAINTY_PAYLOAD.items() if k != "balanced"}
        with pytest.raises(ValidationError):
            UncertaintyResponse(**payload)
