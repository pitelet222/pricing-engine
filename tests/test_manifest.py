"""
Unit tests for src/data/manifest.py — no model artifacts required.

All tests use tmp_path fixtures; no real data/outputs/ is touched.
"""
from __future__ import annotations

import hashlib
import json

from src.data.manifest import check_manifest, write_manifest


class TestWriteManifest:
    def test_returns_manifest_path(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        assert write_manifest(outputs, processed) == outputs / "manifest.json"

    def test_creates_manifest_file(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        write_manifest(outputs, processed)
        assert (outputs / "manifest.json").exists()

    def test_sha256_matches_file_content(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        content = b"col1,col2\n1,2\n"
        (outputs / "demand_model.pkl").write_bytes(content)
        write_manifest(outputs, processed)
        manifest = json.loads((outputs / "manifest.json").read_text())
        expected = hashlib.sha256(content).hexdigest()
        assert manifest["files"]["outputs/demand_model.pkl"]["sha256"] == expected

    def test_size_bytes_recorded(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        (outputs / "forecast_future.csv").write_bytes(b"abc")
        write_manifest(outputs, processed)
        manifest = json.loads((outputs / "manifest.json").read_text())
        assert manifest["files"]["outputs/forecast_future.csv"]["size_bytes"] == 3

    def test_generated_at_present(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        write_manifest(outputs, processed)
        manifest = json.loads((outputs / "manifest.json").read_text())
        assert "generated_at" in manifest

    def test_missing_artifact_not_included(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        # No files written — all artifact entries omitted
        write_manifest(outputs, processed)
        manifest = json.loads((outputs / "manifest.json").read_text())
        assert manifest["files"] == {}

    def test_processed_artifact_included(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        (processed / "avocado_features.csv").write_bytes(b"data")
        write_manifest(outputs, processed)
        manifest = json.loads((outputs / "manifest.json").read_text())
        assert "processed/avocado_features.csv" in manifest["files"]


class TestCheckManifest:
    def test_no_manifest_returns_empty(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        # Backward-compat: no manifest → nothing to check
        assert check_manifest(outputs, processed) == []

    def test_all_matching_returns_empty(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        (outputs / "forecast_future.csv").write_bytes(b"content")
        write_manifest(outputs, processed)
        assert check_manifest(outputs, processed) == []

    def test_missing_artifact_reported(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        (outputs / "forecast_future.csv").write_bytes(b"data")
        write_manifest(outputs, processed)
        (outputs / "forecast_future.csv").unlink()
        issues = check_manifest(outputs, processed)
        assert any("missing" in i.lower() for i in issues)

    def test_stale_artifact_reported(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        path = outputs / "demand_model.pkl"
        path.write_bytes(b"original")
        write_manifest(outputs, processed)
        path.write_bytes(b"regenerated - different checksum")
        issues = check_manifest(outputs, processed)
        assert any("mismatch" in i.lower() or "stale" in i.lower() for i in issues)

    def test_corrupt_manifest_reported(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        (outputs / "manifest.json").write_text("not json {{{", encoding="utf-8")
        issues = check_manifest(outputs, processed)
        assert len(issues) == 1
        assert "manifest" in issues[0].lower()

    def test_malformed_key_reported(self, tmp_path):
        outputs = tmp_path / "outputs"
        processed = tmp_path / "processed"
        outputs.mkdir()
        processed.mkdir()
        bad = {
            "generated_at": "2024-01-01T00:00:00+00:00",
            "files": {"badkey": {"sha256": "abc", "size_bytes": 3}},
        }
        (outputs / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
        issues = check_manifest(outputs, processed)
        assert any("malformed" in i.lower() for i in issues)
