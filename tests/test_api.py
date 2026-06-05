"""
Integration tests for all five API endpoints.

Uses FastAPI's TestClient (synchronous httpx wrapper) so the lifespan runs
and the DataStore is fully populated — no mocking.

Requires model artifacts (data/outputs/). Skipped automatically in CI.
"""

import pytest

pytestmark = pytest.mark.integration

KNOWN_UID = "Albany_conventional"
UNKNOWN_UID = "DOES_NOT_EXIST"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_status_ok(self, client):
        assert client.get("/health").status_code == 200

    def test_status_field(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_series_loaded(self, client):
        data = client.get("/health").json()
        assert data["series_loaded"] == 86

    def test_version_field_present(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)


# ---------------------------------------------------------------------------
# GET /series
# ---------------------------------------------------------------------------


class TestSeries:
    def test_status_ok(self, client):
        assert client.get("/v1/series").status_code == 200

    def test_total_count(self, client):
        data = client.get("/v1/series").json()
        assert data["total"] == 86
        assert len(data["series"]) == 86

    def test_item_fields(self, client):
        item = client.get("/v1/series").json()["series"][0]
        assert "unique_id" in item
        assert "region" in item
        assert item["avocado_type"] in ("organic", "conventional")

    def test_known_series_present(self, client):
        uids = {s["unique_id"] for s in client.get("/v1/series").json()["series"]}
        assert KNOWN_UID in uids


# ---------------------------------------------------------------------------
# GET /forecast/{unique_id}
# ---------------------------------------------------------------------------


class TestForecast:
    def test_status_ok(self, client):
        assert client.get(f"/v1/forecast/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/v1/forecast/{UNKNOWN_UID}").status_code == 404

    def test_twelve_points(self, client):
        data = client.get(f"/v1/forecast/{KNOWN_UID}").json()
        assert len(data["points"]) == 12

    def test_pi_coverage_field(self, client):
        data = client.get(f"/v1/forecast/{KNOWN_UID}").json()
        assert data["pi_coverage"] == pytest.approx(0.90)

    def test_point_fields_present(self, client):
        point = client.get(f"/v1/forecast/{KNOWN_UID}").json()["points"][0]
        for field in (
            "ds",
            "ensemble_weighted",
            "ensemble_lower",
            "ensemble_upper",
            "mstl_ets",
            "mstl_arima",
            "mstl_theta",
            "nhits",
            "seasonal_naive",
        ):
            assert field in point, f"missing field: {field}"

    def test_pi_bounds_ordered(self, client):
        """Lower bound must be <= point forecast <= upper bound for every point."""
        for pt in client.get(f"/v1/forecast/{KNOWN_UID}").json()["points"]:
            assert pt["ensemble_lower"] <= pt["ensemble_weighted"] <= pt["ensemble_upper"]

    def test_prices_positive(self, client):
        for pt in client.get(f"/v1/forecast/{KNOWN_UID}").json()["points"]:
            assert pt["ensemble_weighted"] > 0
            assert pt["nhits"] > 0


# ---------------------------------------------------------------------------
# GET /recommend/{unique_id}
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_status_ok(self, client):
        assert client.get(f"/v1/recommend/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/v1/recommend/{UNKNOWN_UID}").status_code == 404

    def test_required_fields(self, client):
        data = client.get(f"/v1/recommend/{KNOWN_UID}").json()
        for field in (
            "unique_id",
            "is_organic",
            "current_price",
            "optimal_price",
            "price_change_pct",
            "current_revenue",
            "optimal_revenue",
            "revenue_change_pct",
            "elasticity",
        ):
            assert field in data, f"missing field: {field}"

    def test_is_organic_bool(self, client):
        data = client.get(f"/v1/recommend/{KNOWN_UID}").json()
        assert isinstance(data["is_organic"], bool)

    def test_prices_positive(self, client):
        data = client.get(f"/v1/recommend/{KNOWN_UID}").json()
        assert data["current_price"] > 0
        assert data["optimal_price"] > 0

    def test_unique_id_matches(self, client):
        data = client.get(f"/v1/recommend/{KNOWN_UID}").json()
        assert data["unique_id"] == KNOWN_UID


# ---------------------------------------------------------------------------
# GET /explain/{unique_id}
# ---------------------------------------------------------------------------


class TestExplain:
    def test_status_ok(self, client):
        assert client.get(f"/v1/explain/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/v1/explain/{UNKNOWN_UID}").status_code == 404

    def test_three_drivers(self, client):
        data = client.get(f"/v1/explain/{KNOWN_UID}").json()
        assert len(data["drivers"]) == 3

    def test_ranks_are_one_two_three(self, client):
        ranks = [d["driver_rank"] for d in client.get(f"/v1/explain/{KNOWN_UID}").json()["drivers"]]
        assert sorted(ranks) == [1, 2, 3]

    def test_direction_values(self, client):
        valid = {"increases_demand", "decreases_demand"}
        for d in client.get(f"/v1/explain/{KNOWN_UID}").json()["drivers"]:
            assert d["direction"] in valid

    def test_abs_shap_non_negative(self, client):
        for d in client.get(f"/v1/explain/{KNOWN_UID}").json()["drivers"]:
            assert d["abs_shap_value"] >= 0


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_status_ok(self, client):
        r = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 1.25})
        assert r.status_code == 200

    def test_404_unknown_series(self, client):
        r = client.post("/v1/simulate", json={"unique_id": UNKNOWN_UID, "price": 1.25})
        assert r.status_code == 404

    def test_422_negative_price(self, client):
        r = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": -1.0})
        assert r.status_code == 422

    def test_422_zero_price(self, client):
        r = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 0.0})
        assert r.status_code == 422

    def test_response_fields(self, client):
        data = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 1.25}).json()
        for field in (
            "unique_id",
            "price",
            "predicted_log_volume",
            "predicted_volume",
            "current_price",
            "current_volume",
        ):
            assert field in data, f"missing field: {field}"

    def test_price_echoed(self, client):
        data = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 1.50}).json()
        assert data["price"] == pytest.approx(1.50)

    def test_predicted_volume_positive(self, client):
        data = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 1.25}).json()
        assert data["predicted_volume"] > 0

    def test_elasticity_direction(self, client):
        """Higher price should predict lower or equal volume (negative elasticity)."""
        low = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 1.00}).json()
        high = client.post("/v1/simulate", json={"unique_id": KNOWN_UID, "price": 2.00}).json()
        assert high["predicted_volume"] <= low["predicted_volume"]


# ---------------------------------------------------------------------------
# POST /batch-recommend
# ---------------------------------------------------------------------------


class TestBatchRecommend:
    def test_status_ok(self, client):
        r = client.post("/v1/batch-recommend", json={"unique_ids": [KNOWN_UID]})
        assert r.status_code == 200

    def test_single_series(self, client):
        data = client.post("/v1/batch-recommend", json={"unique_ids": [KNOWN_UID]}).json()
        assert data["requested"] == 1
        assert data["found"] == 1
        assert data["not_found"] == []
        assert len(data["results"]) == 1

    def test_unknown_ids_in_not_found(self, client):
        data = client.post(
            "/v1/batch-recommend",
            json={"unique_ids": [KNOWN_UID, UNKNOWN_UID]},
        ).json()
        assert data["requested"] == 2
        assert data["found"] == 1
        assert UNKNOWN_UID in data["not_found"]
        assert len(data["results"]) == 1

    def test_empty_list_rejected(self, client):
        r = client.post("/v1/batch-recommend", json={"unique_ids": []})
        assert r.status_code == 422

    def test_results_match_individual(self, client):
        batch = client.post("/v1/batch-recommend", json={"unique_ids": [KNOWN_UID]}).json()
        individual = client.get(f"/v1/recommend/{KNOWN_UID}").json()
        assert batch["results"][0]["optimal_price"] == pytest.approx(individual["optimal_price"])


# ---------------------------------------------------------------------------
# X-Request-ID header tracing
# ---------------------------------------------------------------------------


class TestRequestId:
    def test_x_request_id_in_response_headers(self, client):
        r = client.get("/health")
        assert "x-request-id" in r.headers

    def test_request_id_is_8_chars(self, client):
        r = client.get("/health")
        assert len(r.headers["x-request-id"]) == 8

    def test_different_requests_get_different_ids(self, client):
        id1 = client.get("/health").headers["x-request-id"]
        id2 = client.get("/health").headers["x-request-id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# GET /metrics (Prometheus)
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_endpoint_ok(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_content_type(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_contains_request_count(self, client):
        client.get("/health")  # ensure at least one request is recorded
        r = client.get("/metrics")
        assert "api_requests_total" in r.text


# ---------------------------------------------------------------------------
# GET /uncertainty/{unique_id}
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_status_ok(self, client):
        assert client.get(f"/v1/uncertainty/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/v1/uncertainty/{UNKNOWN_UID}").status_code == 404

    def test_required_fields(self, client):
        data = client.get(f"/v1/uncertainty/{KNOWN_UID}").json()
        for field in (
            "unique_id",
            "is_organic",
            "current_price",
            "conservative",
            "balanced",
            "aggressive",
            "rev_p10",
            "rev_p50",
            "rev_p90",
            "rev_spread_pct",
            "downside_risk_pct",
            "uplift_mean",
            "uplift_std",
            "uplift_sharpe",
            "strategies_agree",
            "price_direction_conservative",
            "price_direction_aggressive",
        ):
            assert field in data, f"missing field: {field}"

    def test_strategy_fields(self, client):
        data = client.get(f"/v1/uncertainty/{KNOWN_UID}").json()
        for strategy in ("conservative", "balanced", "aggressive"):
            assert "opt_price" in data[strategy]
            assert "rev_uplift_pct" in data[strategy]
            assert data[strategy]["opt_price"] > 0

    def test_prices_positive(self, client):
        data = client.get(f"/v1/uncertainty/{KNOWN_UID}").json()
        assert data["current_price"] > 0

    def test_strategies_agree_is_bool(self, client):
        data = client.get(f"/v1/uncertainty/{KNOWN_UID}").json()
        assert isinstance(data["strategies_agree"], bool)

    def test_direction_values_valid(self, client):
        data = client.get(f"/v1/uncertainty/{KNOWN_UID}").json()
        assert data["price_direction_conservative"] in (1, -1)
        assert data["price_direction_aggressive"] in (1, -1)

    def test_percentiles_ordered(self, client):
        data = client.get(f"/v1/uncertainty/{KNOWN_UID}").json()
        assert data["rev_p10"] <= data["rev_p50"] <= data["rev_p90"]


# ---------------------------------------------------------------------------
# GET /series — pagination
# ---------------------------------------------------------------------------


class TestSeriesPagination:
    def test_default_returns_all(self, client):
        data = client.get("/v1/series").json()
        assert data["total"] == 86
        assert len(data["series"]) == 86
        assert data["limit"] == 86
        assert data["offset"] == 0

    def test_custom_limit(self, client):
        data = client.get("/v1/series?limit=5").json()
        assert len(data["series"]) == 5
        assert data["total"] == 86
        assert data["limit"] == 5

    def test_custom_offset(self, client):
        data = client.get("/v1/series?limit=86&offset=80").json()
        assert len(data["series"]) == 6  # 86 - 80

    def test_offset_beyond_total_returns_empty(self, client):
        data = client.get("/v1/series?offset=100").json()
        assert data["series"] == []
        assert data["total"] == 86

    def test_limit_and_offset_return_different_pages(self, client):
        first = client.get("/v1/series?limit=10&offset=0").json()["series"]
        second = client.get("/v1/series?limit=10&offset=10").json()["series"]
        first_ids = {s["unique_id"] for s in first}
        second_ids = {s["unique_id"] for s in second}
        assert first_ids.isdisjoint(second_ids)

    def test_limit_above_86_rejected(self, client):
        assert client.get("/v1/series?limit=87").status_code == 422

    def test_limit_zero_rejected(self, client):
        assert client.get("/v1/series?limit=0").status_code == 422
