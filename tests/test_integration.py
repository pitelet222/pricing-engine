"""
test_integration.py — End-to-end API tests backed by a synthetic toy DataStore.

All DataStore artefacts (LightGBM model, CSVs, label encoder) are built from
scratch at test-session start using the src pipeline modules.  No notebook runs
and no pre-existing data/outputs/ directory are needed.

The entire module is auto-skipped when lightgbm or scikit-learn is absent so
the standard unit-test job (which omits heavy ML packages) is not affected.
The dedicated CI integration job installs these packages explicitly.

Toy DataStore characteristics
------------------------------
Series  : 4 (Albany × 2 types, Houston × 2 types)
Weeks   : 60 per series  →  ~56 rows per series after warmup dropout
Model   : LightGBM regression, 30 rounds (fast, non-production accuracy)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.features import build_features
from src.models.pricer import (
    FEATURE_COLS,
    TARGET_COL,
    compute_elasticity,
    optimize_price,
)
from src.models.uncertainty import (
    BASE_QUANTILE_PARAMS,
    QUANTILES,
    compute_conformal_pi,
    compute_risk_metrics,
    optimize_strategies,
)

# Auto-skip the entire module if heavy deps are not installed.
# Placed after src imports (which have no heavy ML at module level) so that
# isort/ruff does not flag the src imports as E402.
lgb = pytest.importorskip("lightgbm")
_sklearn_prep = pytest.importorskip("sklearn.preprocessing")


# ---------------------------------------------------------------------------
# Toy raw data
# ---------------------------------------------------------------------------

def _make_toy_clean_df(n_weeks: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Return a minimal cleaned avocado DataFrame: 2 regions × 2 types × n_weeks.

    Mirrors the schema produced by preprocessing.clean_data so that
    build_features() can consume it directly.
    """
    rng = np.random.default_rng(seed)
    series_specs = [
        ("Albany",  1, 1.80,  50_000),
        ("Albany",  0, 1.20,  80_000),
        ("Houston", 1, 1.90, 100_000),
        ("Houston", 0, 1.15, 200_000),
    ]
    start = pd.Timestamp("2015-01-04")
    rows = []
    for region, is_organic, base_price, base_vol in series_specs:
        prices = np.clip(base_price + rng.normal(0, 0.15, n_weeks), 0.50, 3.50)
        for i, price in enumerate(prices):
            vol = max(1_000, int(base_vol * np.exp(-0.3 * price + rng.normal(0, 0.10))))
            d = rng.dirichlet([2.0, 3.0, 1.0])
            p4046, p4225, p4770 = vol * d[0], vol * d[1], vol * d[2]
            bags = vol * 0.30
            rows.append({
                "Date":         start + pd.Timedelta(weeks=i),
                "AveragePrice": round(float(price), 2),
                "Total Volume": float(vol),
                "4046":         round(p4046),
                "4225":         round(p4225),
                "4770":         round(p4770),
                "Total Bags":   round(bags),
                "Small Bags":   round(bags * 0.60),
                "Large Bags":   round(bags * 0.35),
                "XLarge Bags":  round(bags * 0.05),
                "region":       region,
                "is_organic":   is_organic,
            })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Artefact builder
# ---------------------------------------------------------------------------

def _build_toy_artifacts(outputs_dir: Path, processed_dir: Path) -> None:
    """
    Write all 8 DataStore artefacts to outputs_dir / processed_dir.

    Uses the src pipeline modules and a fast 30-round LightGBM model so the
    whole fixture completes in a few seconds.
    """
    import joblib
    from sklearn.preprocessing import LabelEncoder

    # 1. Feature engineering -------------------------------------------------
    clean_df = _make_toy_clean_df()
    feat_df = build_features(clean_df)

    # 2. Label-encode regions ------------------------------------------------
    le = LabelEncoder().fit(sorted(feat_df["region"].unique()))
    feat_df = feat_df.copy()
    feat_df["region_encoded"] = le.transform(feat_df["region"])
    feat_df["unique_id"] = (
        feat_df["region"]
        + "_"
        + feat_df["is_organic"].map({1: "organic", 0: "conventional"})
    )

    # Temporal sort used for train split and for writing processed CSV.
    feat_df = feat_df.sort_values("Date").reset_index(drop=True)

    # 3. Train LightGBM demand model (fast 30 rounds) -----------------------
    n_train = int(len(feat_df) * 0.80)
    train_df = feat_df.iloc[:n_train]
    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df[TARGET_COL].astype(float)

    lgb_params = {
        "objective":      "regression",
        "metric":         "rmse",
        "learning_rate":  0.10,
        "num_leaves":     7,
        "min_child_samples": 5,
        "n_jobs":         1,
        "verbose":        -1,
        "random_state":   42,
    }
    model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), num_boost_round=30)

    model_bundle = {"model": model, "features": FEATURE_COLS, "target": TARGET_COL}
    with open(outputs_dir / "demand_model.pkl", "wb") as fh:
        pickle.dump(model_bundle, fh)
    joblib.dump(le, outputs_dir / "region_label_encoder.pkl")

    # 4. Processed features CSV (without unique_id / region_encoded —
    #    the loader re-derives these at startup via _build_latest_ctx) -------
    feat_df.drop(columns=["unique_id", "region_encoded"], errors="ignore").to_csv(
        processed_dir / "avocado_features.csv", index=False
    )

    # 5. Pricing recommendations ---------------------------------------------
    series_ids = sorted(feat_df["unique_id"].unique())
    rec_rows = []
    for uid in series_ids:
        sub = feat_df[feat_df["unique_id"] == uid]
        p_mean = float(sub["AveragePrice"].mean())
        latest = sub.iloc[-1]
        res = optimize_price(latest, p_mean, model, FEATURE_COLS)
        curr_r = res["current_revenue"]
        opt_r  = res["optimal_revenue"]
        rev_pct  = (opt_r - curr_r) / curr_r * 100 if curr_r > 0 else 0.0
        price_pct = (
            (res["optimal_price"] - res["current_price"]) / res["current_price"] * 100
        )
        rec_rows.append({
            "unique_id":         uid,
            "is_organic":        int(sub["is_organic"].iloc[0]),
            "current_price":     res["current_price"],
            "optimal_price":     res["optimal_price"],
            "price_change_pct":  round(price_pct, 2),
            "current_revenue":   res["current_revenue"],
            "optimal_revenue":   res["optimal_revenue"],
            "revenue_change_pct": round(rev_pct, 2),
            "elasticity":        round(compute_elasticity(latest, model, FEATURE_COLS), 4),
        })
    pd.DataFrame(rec_rows).to_csv(outputs_dir / "pricing_recommendations.csv", index=False)

    # 6. Conformal PI (from in-sample residuals) -----------------------------
    cv_rows = []
    for uid in series_ids:
        sub = feat_df[feat_df["unique_id"] == uid]
        preds = model.predict(sub[FEATURE_COLS].astype(float))
        for y_val, f_val in zip(sub[TARGET_COL].values, preds):
            cv_rows.append({"unique_id": uid, "y": float(y_val), "forecast": float(f_val)})
    pi_df = compute_conformal_pi(pd.DataFrame(cv_rows), model_col="forecast")
    pi_df.to_csv(outputs_dir / "conformal_pi_stats.csv", index=False)

    # 7. Uncertainty pricing (train 3 quantile models, 30 rounds each) ------
    quant_models: dict = {}
    for q in QUANTILES:
        q_params = {
            **BASE_QUANTILE_PARAMS,
            "alpha":             q,
            "num_leaves":        7,
            "min_child_samples": 5,
            "n_jobs":            1,
            "random_state":      42,
        }
        quant_models[q] = lgb.train(
            q_params, lgb.Dataset(X_train, label=y_train), num_boost_round=30
        )

    unc_rows = []
    for uid in series_ids:
        sub = feat_df[feat_df["unique_id"] == uid]
        p_mean = float(sub["AveragePrice"].mean())
        latest = sub.iloc[-1]
        res = optimize_strategies(latest, p_mean, quant_models, FEATURE_COLS)
        res["unique_id"] = uid
        res["is_organic"] = int(sub["is_organic"].iloc[0])
        unc_rows.append(res)
    unc_df = compute_risk_metrics(pd.DataFrame(unc_rows))
    unc_df.to_csv(outputs_dir / "uncertainty_pricing.csv", index=False)

    # 8. Forecast future (12 synthetic weeks per series) --------------------
    last_date = feat_df["Date"].max()
    forecast_rows = []
    for uid in series_ids:
        sub = feat_df[feat_df["unique_id"] == uid]
        latest = sub.iloc[-1]
        pred = float(
            model.predict(latest[FEATURE_COLS].to_frame().T.astype(float))[0]
        )
        for h in range(1, 13):
            forecast_rows.append({
                "unique_id":        uid,
                "ds":               (last_date + pd.Timedelta(weeks=h)).date(),
                "Ensemble_weighted": round(pred, 4),
                "MSTL_ETS":         round(pred * 1.001, 4),
                "MSTL_ARIMA":       round(pred * 0.999, 4),
                "MSTL_Theta":       round(pred * 1.002, 4),
                "NHITS":            round(pred * 0.998, 4),
                "SeasonalNaive":    round(pred * 1.003, 4),
            })
    pd.DataFrame(forecast_rows).to_csv(outputs_dir / "forecast_future.csv", index=False)

    # 9. SHAP-like drivers (proxy: top-3 features by LightGBM gain) --------
    gains = model.feature_importance(importance_type="gain")
    max_gain = float(gains.max()) or 1.0
    top3_idx = np.argsort(gains)[-3:][::-1]

    shap_rows = []
    for uid in series_ids:
        sub = feat_df[feat_df["unique_id"] == uid]
        latest = sub.iloc[-1]
        for rank, feat_idx in enumerate(top3_idx, start=1):
            feat_name = FEATURE_COLS[feat_idx]
            shap_val = round(float(gains[feat_idx]) / max_gain * 0.50, 6)
            shap_rows.append({
                "unique_id":     uid,
                "driver_rank":   rank,
                "feature":       feat_name,
                "shap_value":    shap_val,
                "abs_shap_value": abs(shap_val),
                "direction":     "increases_demand",
                "feature_value": round(float(latest[feat_name]), 6),
            })
    pd.DataFrame(shap_rows).to_csv(outputs_dir / "shap_top_drivers.csv", index=False)


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def integration_client(tmp_path_factory):
    """
    Session-scoped TestClient backed by a toy DataStore built from src modules.

    Monkeypatches loader._OUTPUTS and loader._PROCESSED so the lifespan loads
    artefacts from the temp directory rather than data/outputs/.
    """
    import src.api.routes as _routes
    import src.data.loader as _loader

    tmp = tmp_path_factory.mktemp("toy_ds")
    outputs_dir = tmp / "outputs"
    processed_dir = tmp / "processed"
    outputs_dir.mkdir()
    processed_dir.mkdir()

    _build_toy_artifacts(outputs_dir, processed_dir)

    # Clear the route-level response cache before booting so stale entries
    # from other fixtures (e.g. a real-DataStore client) cannot bleed through.
    _routes._CACHE.clear()

    with (
        patch.object(_loader, "_OUTPUTS", outputs_dir),
        patch.object(_loader, "_PROCESSED", processed_dir),
    ):
        from src.api.main import app
        with TestClient(app) as client:
            yield client

    _routes._CACHE.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_UID = "Albany_organic"
_ALL_UIDS = {"Albany_organic", "Albany_conventional", "Houston_organic", "Houston_conventional"}


class TestHealth:
    def test_status_ok(self, integration_client):
        r = integration_client.get("/health")
        assert r.status_code == 200

    def test_status_field(self, integration_client):
        assert integration_client.get("/health").json()["status"] == "ok"

    def test_series_loaded(self, integration_client):
        assert integration_client.get("/health").json()["series_loaded"] == 4

    def test_version_present(self, integration_client):
        body = integration_client.get("/health").json()
        assert "version" in body and body["version"]

    def test_manifest_ok_present(self, integration_client):
        body = integration_client.get("/health").json()
        assert "manifest_ok" in body
        assert isinstance(body["manifest_ok"], bool)

    def test_artifacts_generated_at_present(self, integration_client):
        body = integration_client.get("/health").json()
        assert "artifacts_generated_at" in body  # value is None when no manifest


class TestSeries:
    def test_status_ok(self, integration_client):
        assert integration_client.get("/v1/series").status_code == 200

    def test_total_four(self, integration_client):
        assert integration_client.get("/v1/series").json()["total"] == 4

    def test_all_series_present(self, integration_client):
        ids = {s["unique_id"] for s in integration_client.get("/v1/series").json()["series"]}
        assert ids == _ALL_UIDS

    def test_avocado_type_field(self, integration_client):
        series = integration_client.get("/v1/series").json()["series"]
        for s in series:
            assert s["avocado_type"] in ("organic", "conventional")


class TestForecast:
    def test_status_ok(self, integration_client):
        assert integration_client.get(f"/v1/forecast/{_UID}").status_code == 200

    def test_twelve_points(self, integration_client):
        body = integration_client.get(f"/v1/forecast/{_UID}").json()
        assert len(body["points"]) == 12

    def test_unique_id_echo(self, integration_client):
        assert integration_client.get(f"/v1/forecast/{_UID}").json()["unique_id"] == _UID

    def test_pi_ordering(self, integration_client):
        pt = integration_client.get(f"/v1/forecast/{_UID}").json()["points"][0]
        assert pt["ensemble_lower"] <= pt["ensemble_weighted"] <= pt["ensemble_upper"]

    def test_unknown_series_404(self, integration_client):
        assert integration_client.get("/v1/forecast/DoesNotExist_organic").status_code == 404


class TestRecommend:
    def test_status_ok(self, integration_client):
        assert integration_client.get(f"/v1/recommend/{_UID}").status_code == 200

    def test_positive_optimal_price(self, integration_client):
        body = integration_client.get(f"/v1/recommend/{_UID}").json()
        assert body["optimal_price"] > 0

    def test_required_fields(self, integration_client):
        body = integration_client.get(f"/v1/recommend/{_UID}").json()
        for field in ("current_price", "optimal_price", "revenue_change_pct", "elasticity"):
            assert field in body

    def test_unknown_series_404(self, integration_client):
        assert integration_client.get("/v1/recommend/DoesNotExist_organic").status_code == 404


class TestBatchRecommend:
    def test_status_ok(self, integration_client):
        r = integration_client.post(
            "/v1/batch-recommend",
            json={"unique_ids": [_UID, "Houston_conventional"]},
        )
        assert r.status_code == 200

    def test_found_count(self, integration_client):
        r = integration_client.post(
            "/v1/batch-recommend",
            json={"unique_ids": [_UID, "Houston_conventional", "Ghost_organic"]},
        )
        body = r.json()
        assert body["requested"] == 3
        assert body["found"] == 2
        assert body["not_found"] == ["Ghost_organic"]


class TestExplain:
    def test_status_ok(self, integration_client):
        assert integration_client.get(f"/v1/explain/{_UID}").status_code == 200

    def test_three_drivers(self, integration_client):
        body = integration_client.get(f"/v1/explain/{_UID}").json()
        assert len(body["drivers"]) == 3

    def test_driver_ranks(self, integration_client):
        drivers = integration_client.get(f"/v1/explain/{_UID}").json()["drivers"]
        assert [d["driver_rank"] for d in drivers] == [1, 2, 3]

    def test_driver_fields(self, integration_client):
        driver = integration_client.get(f"/v1/explain/{_UID}").json()["drivers"][0]
        for field in ("feature", "shap_value", "abs_shap_value", "direction"):
            assert field in driver

    def test_unknown_series_404(self, integration_client):
        assert integration_client.get("/v1/explain/DoesNotExist_organic").status_code == 404


class TestUncertainty:
    def test_status_ok(self, integration_client):
        assert integration_client.get(f"/v1/uncertainty/{_UID}").status_code == 200

    def test_strategies_present(self, integration_client):
        body = integration_client.get(f"/v1/uncertainty/{_UID}").json()
        for strategy in ("conservative", "balanced", "aggressive"):
            assert strategy in body
            assert body[strategy]["opt_price"] > 0

    def test_risk_fields_present(self, integration_client):
        body = integration_client.get(f"/v1/uncertainty/{_UID}").json()
        for field in ("downside_risk_pct", "uplift_sharpe", "strategies_agree"):
            assert field in body

    def test_pi_ordering(self, integration_client):
        body = integration_client.get(f"/v1/uncertainty/{_UID}").json()
        assert body["rev_p10"] <= body["rev_p50"] <= body["rev_p90"]

    def test_unknown_series_404(self, integration_client):
        assert integration_client.get("/v1/uncertainty/DoesNotExist_organic").status_code == 404


class TestSimulate:
    def test_status_ok(self, integration_client):
        r = integration_client.post("/v1/simulate", json={"unique_id": _UID, "price": 1.50})
        assert r.status_code == 200

    def test_response_fields(self, integration_client):
        body = integration_client.post(
            "/v1/simulate", json={"unique_id": _UID, "price": 1.50}
        ).json()
        for field in ("unique_id", "price", "predicted_volume", "predicted_log_volume"):
            assert field in body

    def test_positive_volume(self, integration_client):
        body = integration_client.post(
            "/v1/simulate", json={"unique_id": _UID, "price": 1.50}
        ).json()
        assert body["predicted_volume"] > 0

    def test_price_echoed(self, integration_client):
        body = integration_client.post(
            "/v1/simulate", json={"unique_id": _UID, "price": 2.25}
        ).json()
        assert body["price"] == pytest.approx(2.25)

    def test_unknown_series_404(self, integration_client):
        r = integration_client.post(
            "/v1/simulate", json={"unique_id": "DoesNotExist_organic", "price": 1.50}
        )
        assert r.status_code == 404
