"""
Unit tests for DataStore helpers in src/data/loader.py — no model artifacts required.

These tests always run in CI. They verify the pure helper functions that
do not touch the filesystem at runtime, plus the DataStore field contract
so changes to the dataclass structure are caught immediately.

Coverage:
  - _require: FileNotFoundError for missing paths, passthrough for existing paths
  - _build_series_meta: dict construction, region extraction, avocado_type assignment
  - DataStore: field names match the documented contract
"""
from __future__ import annotations

import dataclasses
import pathlib

import pandas as pd
import pytest

from src.data.loader import DataStore, _build_series_meta, _require

# ---------------------------------------------------------------------------
# _require
# ---------------------------------------------------------------------------

class TestRequire:
    def test_raises_for_missing_path(self, tmp_path):
        missing = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError, match="Required artefact not found"):
            _require(missing)

    def test_error_message_includes_path(self, tmp_path):
        missing = tmp_path / "conformal_pi_stats.csv"
        with pytest.raises(FileNotFoundError, match="conformal_pi_stats.csv"):
            _require(missing)

    def test_error_message_mentions_notebooks(self, tmp_path):
        """Operator hint: 'Run the notebooks...' must appear in the error."""
        missing = tmp_path / "demand_model.pkl"
        with pytest.raises(FileNotFoundError, match="notebooks"):
            _require(missing)

    def test_returns_path_for_existing_file(self, tmp_path):
        existing = tmp_path / "artifact.pkl"
        existing.touch()
        result = _require(existing)
        assert result == existing

    def test_return_type_is_path(self, tmp_path):
        existing = tmp_path / "artifact.csv"
        existing.touch()
        assert isinstance(_require(existing), pathlib.Path)


# ---------------------------------------------------------------------------
# _build_series_meta
# ---------------------------------------------------------------------------

_RECS_DF = pd.DataFrame({
    "unique_id": [
        "Albany_conventional",
        "Albany_organic",
        "Denver_conventional",
    ],
    "is_organic": [0, 1, 0],
})


class TestBuildSeriesMeta:
    def test_returns_dict(self):
        assert isinstance(_build_series_meta(_RECS_DF), dict)

    def test_all_series_are_keys(self):
        meta = _build_series_meta(_RECS_DF)
        assert set(meta.keys()) == {
            "Albany_conventional",
            "Albany_organic",
            "Denver_conventional",
        }

    def test_conventional_type(self):
        meta = _build_series_meta(_RECS_DF)
        assert meta["Albany_conventional"]["avocado_type"] == "conventional"

    def test_organic_type(self):
        meta = _build_series_meta(_RECS_DF)
        assert meta["Albany_organic"]["avocado_type"] == "organic"

    def test_region_extracted_from_uid(self):
        meta = _build_series_meta(_RECS_DF)
        assert meta["Albany_conventional"]["region"] == "Albany"
        assert meta["Albany_organic"]["region"] == "Albany"
        assert meta["Denver_conventional"]["region"] == "Denver"

    def test_each_value_has_region_and_type_keys(self):
        meta = _build_series_meta(_RECS_DF)
        for uid, value in meta.items():
            assert "region" in value, f"missing 'region' for {uid}"
            assert "avocado_type" in value, f"missing 'avocado_type' for {uid}"

    def test_multi_word_region_name(self):
        """rsplit('_', 1) must split on the LAST underscore only."""
        df = pd.DataFrame({
            "unique_id": ["LosAngeles_conventional", "SanFrancisco_organic"],
            "is_organic": [0, 1],
        })
        meta = _build_series_meta(df)
        assert meta["LosAngeles_conventional"]["region"] == "LosAngeles"
        assert meta["SanFrancisco_organic"]["region"] == "SanFrancisco"

    def test_empty_dataframe_returns_empty_dict(self):
        df = pd.DataFrame({"unique_id": [], "is_organic": []})
        assert _build_series_meta(df) == {}

    def test_is_organic_as_integer_string_coercion(self):
        """is_organic values are coerced via int() so float 1.0 counts as organic."""
        df = pd.DataFrame({
            "unique_id": ["Albany_organic"],
            "is_organic": [1.0],
        })
        meta = _build_series_meta(df)
        assert meta["Albany_organic"]["avocado_type"] == "organic"


# ---------------------------------------------------------------------------
# DataStore field contract
# ---------------------------------------------------------------------------

class TestDataStoreContract:
    def test_has_expected_field_names(self):
        """Catches accidental renames or deletions of DataStore attributes."""
        fields = {f.name for f in dataclasses.fields(DataStore)}
        expected = {
            "forecast_df",
            "pi_df",
            "recommendations_df",
            "uncertainty_df",
            "shap_drivers_df",
            "latest_ctx_df",
            "model_bundle",
            "series_meta",
        }
        assert fields == expected

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DataStore)

    def test_field_count(self):
        assert len(dataclasses.fields(DataStore)) == 8
