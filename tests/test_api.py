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
        assert client.get("/series").status_code == 200

    def test_total_count(self, client):
        data = client.get("/series").json()
        assert data["total"] == 86
        assert len(data["series"]) == 86

    def test_item_fields(self, client):
        item = client.get("/series").json()["series"][0]
        assert "unique_id" in item
        assert "region" in item
        assert item["avocado_type"] in ("organic", "conventional")

    def test_known_series_present(self, client):
        uids = {s["unique_id"] for s in client.get("/series").json()["series"]}
        assert KNOWN_UID in uids


# ---------------------------------------------------------------------------
# GET /forecast/{unique_id}
# ---------------------------------------------------------------------------

class TestForecast:
    def test_status_ok(self, client):
        assert client.get(f"/forecast/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/forecast/{UNKNOWN_UID}").status_code == 404

    def test_twelve_points(self, client):
        data = client.get(f"/forecast/{KNOWN_UID}").json()
        assert len(data["points"]) == 12

    def test_pi_coverage_field(self, client):
        data = client.get(f"/forecast/{KNOWN_UID}").json()
        assert data["pi_coverage"] == pytest.approx(0.90)

    def test_point_fields_present(self, client):
        point = client.get(f"/forecast/{KNOWN_UID}").json()["points"][0]
        for field in ("ds", "ensemble_weighted", "ensemble_lower", "ensemble_upper",
                      "mstl_ets", "mstl_arima", "mstl_theta", "nhits", "seasonal_naive"):
            assert field in point, f"missing field: {field}"

    def test_pi_bounds_ordered(self, client):
        """Lower bound must be <= point forecast <= upper bound for every point."""
        for pt in client.get(f"/forecast/{KNOWN_UID}").json()["points"]:
            assert pt["ensemble_lower"] <= pt["ensemble_weighted"] <= pt["ensemble_upper"]

    def test_prices_positive(self, client):
        for pt in client.get(f"/forecast/{KNOWN_UID}").json()["points"]:
            assert pt["ensemble_weighted"] > 0
            assert pt["nhits"] > 0


# ---------------------------------------------------------------------------
# GET /recommend/{unique_id}
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_status_ok(self, client):
        assert client.get(f"/recommend/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/recommend/{UNKNOWN_UID}").status_code == 404

    def test_required_fields(self, client):
        data = client.get(f"/recommend/{KNOWN_UID}").json()
        for field in ("unique_id", "is_organic", "current_price", "optimal_price",
                      "price_change_pct", "current_revenue", "optimal_revenue",
                      "revenue_change_pct", "elasticity"):
            assert field in data, f"missing field: {field}"

    def test_is_organic_bool(self, client):
        data = client.get(f"/recommend/{KNOWN_UID}").json()
        assert isinstance(data["is_organic"], bool)

    def test_prices_positive(self, client):
        data = client.get(f"/recommend/{KNOWN_UID}").json()
        assert data["current_price"] > 0
        assert data["optimal_price"] > 0

    def test_unique_id_matches(self, client):
        data = client.get(f"/recommend/{KNOWN_UID}").json()
        assert data["unique_id"] == KNOWN_UID


# ---------------------------------------------------------------------------
# GET /explain/{unique_id}
# ---------------------------------------------------------------------------

class TestExplain:
    def test_status_ok(self, client):
        assert client.get(f"/explain/{KNOWN_UID}").status_code == 200

    def test_404_unknown(self, client):
        assert client.get(f"/explain/{UNKNOWN_UID}").status_code == 404

    def test_three_drivers(self, client):
        data = client.get(f"/explain/{KNOWN_UID}").json()
        assert len(data["drivers"]) == 3

    def test_ranks_are_one_two_three(self, client):
        ranks = [d["driver_rank"] for d in client.get(f"/explain/{KNOWN_UID}").json()["drivers"]]
        assert sorted(ranks) == [1, 2, 3]

    def test_direction_values(self, client):
        valid = {"increases_demand", "decreases_demand"}
        for d in client.get(f"/explain/{KNOWN_UID}").json()["drivers"]:
            assert d["direction"] in valid

    def test_abs_shap_non_negative(self, client):
        for d in client.get(f"/explain/{KNOWN_UID}").json()["drivers"]:
            assert d["abs_shap_value"] >= 0


# ---------------------------------------------------------------------------
# POST /simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_status_ok(self, client):
        r = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 1.25})
        assert r.status_code == 200

    def test_404_unknown_series(self, client):
        r = client.post("/simulate", json={"unique_id": UNKNOWN_UID, "price": 1.25})
        assert r.status_code == 404

    def test_422_negative_price(self, client):
        r = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": -1.0})
        assert r.status_code == 422

    def test_422_zero_price(self, client):
        r = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 0.0})
        assert r.status_code == 422

    def test_response_fields(self, client):
        data = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 1.25}).json()
        for field in ("unique_id", "price", "predicted_log_volume",
                      "predicted_volume", "current_price", "current_volume"):
            assert field in data, f"missing field: {field}"

    def test_price_echoed(self, client):
        data = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 1.50}).json()
        assert data["price"] == pytest.approx(1.50)

    def test_predicted_volume_positive(self, client):
        data = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 1.25}).json()
        assert data["predicted_volume"] > 0

    def test_elasticity_direction(self, client):
        """Higher price should predict lower or equal volume (negative elasticity)."""
        low = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 1.00}).json()
        high = client.post("/simulate", json={"unique_id": KNOWN_UID, "price": 2.00}).json()
        assert high["predicted_volume"] <= low["predicted_volume"]
