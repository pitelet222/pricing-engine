"""
Unit tests for DataStore: verifies all artifacts load correctly and have the
expected shape and content — independently of the API layer.

Requires model artifacts (data/outputs/). Skipped automatically in CI.
"""
import pytest

from src.data.loader import DataStore, load_datastore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ds() -> DataStore:
    return load_datastore()


class TestDataStoreLoads:
    def test_returns_datastore_instance(self, ds):
        assert isinstance(ds, DataStore)

    def test_series_meta_count(self, ds):
        assert len(ds.series_meta) == 86

    def test_series_meta_fields(self, ds):
        uid = next(iter(ds.series_meta))
        meta = ds.series_meta[uid]
        assert "region" in meta
        assert meta["avocado_type"] in ("organic", "conventional")


class TestForecastDf:
    def test_shape(self, ds):
        # 86 series × 12 weeks; 12 columns including index columns, models, ensembles
        assert ds.forecast_df.shape == (1032, 12)

    def test_required_columns(self, ds):
        for col in ("unique_id", "ds", "Ensemble_weighted",
                    "MSTL_ETS", "MSTL_ARIMA", "MSTL_Theta",
                    "NHITS", "SeasonalNaive"):
            assert col in ds.forecast_df.columns, f"missing column: {col}"

    def test_no_nulls_in_ensemble(self, ds):
        for col in ("Ensemble_weighted", "MSTL_ETS", "MSTL_ARIMA",
                    "MSTL_Theta", "NHITS", "SeasonalNaive"):
            assert ds.forecast_df[col].isna().sum() == 0, f"{col} has nulls"


class TestPiDf:
    def test_shape(self, ds):
        assert ds.pi_df.shape == (86, 7)

    def test_lower_q_less_than_upper_q(self, ds):
        assert (ds.pi_df["lower_q"] <= ds.pi_df["upper_q"]).all()


class TestRecommendationsDf:
    def test_shape(self, ds):
        # 86 series; 13 columns (9 base + 4 forecast-context columns from nb04 §8.1)
        assert ds.recommendations_df.shape == (86, 13)

    def test_prices_positive(self, ds):
        assert (ds.recommendations_df["current_price"] > 0).all()
        assert (ds.recommendations_df["optimal_price"] > 0).all()


class TestShapDriversDf:
    def test_shape(self, ds):
        # 86 series × top 3 drivers each
        assert ds.shap_drivers_df.shape == (258, 7)

    def test_ranks_are_one_to_three(self, ds):
        assert set(ds.shap_drivers_df["driver_rank"].unique()) == {1, 2, 3}

    def test_direction_values(self, ds):
        valid = {"increases_demand", "decreases_demand"}
        assert set(ds.shap_drivers_df["direction"].unique()).issubset(valid)


class TestLatestCtxDf:
    def test_one_row_per_series(self, ds):
        assert len(ds.latest_ctx_df) == 86
        assert ds.latest_ctx_df["unique_id"].nunique() == 86

    def test_has_region_encoded(self, ds):
        assert "region_encoded" in ds.latest_ctx_df.columns

    def test_all_model_features_present(self, ds):
        features = ds.model_bundle["features"]
        missing = [f for f in features if f not in ds.latest_ctx_df.columns]
        assert missing == [], f"missing features in latest_ctx_df: {missing}"


class TestModelBundle:
    def test_keys(self, ds):
        assert set(ds.model_bundle.keys()) == {"model", "features", "target"}

    def test_target(self, ds):
        assert ds.model_bundle["target"] == "log_total_volume"

    def test_features_count(self, ds):
        assert len(ds.model_bundle["features"]) == 26

    def test_model_can_predict(self, ds):
        import pandas as pd
        row = ds.latest_ctx_df[
            ds.latest_ctx_df["unique_id"] == "Albany_conventional"
        ].iloc[0]
        X = row[ds.model_bundle["features"]].to_frame().T.astype(float)
        pred = ds.model_bundle["model"].predict(X)
        assert len(pred) == 1
        assert pred[0] > 0
