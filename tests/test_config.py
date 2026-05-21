"""
Unit tests for application settings — no model artifacts required.

These tests always run in CI. They verify that Settings loads with sensible
defaults and that env var overrides work correctly.
"""
import asyncio
import json
import logging

import pytest
from fastapi import HTTPException


class TestSettingsDefaults:
    def test_api_version(self):
        from src.config import settings
        assert settings.api_version == "0.1.0"

    def test_log_level_uppercase(self):
        from src.config import settings
        assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        assert settings.log_level == settings.log_level.upper()

    def test_outputs_dir_name(self):
        from src.config import settings
        assert settings.outputs_dir.name == "outputs"

    def test_processed_dir_name(self):
        from src.config import settings
        assert settings.processed_dir.name == "processed"

    def test_outputs_is_child_of_data_root(self):
        from src.config import settings
        assert settings.outputs_dir.parent == settings.data_root

    def test_api_title_non_empty(self):
        from src.config import settings
        assert len(settings.api_title) > 0

    def test_api_description_non_empty(self):
        from src.config import settings
        assert len(settings.api_description) > 0


class TestSettingsEnvOverride:
    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("PRICING_LOG_LEVEL", "debug")
        # Re-import to pick up env change (module-level singleton won't re-evaluate,
        # so we instantiate a fresh Settings directly).
        from src.config import Settings
        s = Settings()
        assert s.log_level == "DEBUG"  # validator upcases

    def test_data_root_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PRICING_DATA_ROOT", str(tmp_path))
        from src.config import Settings
        s = Settings()
        assert s.data_root == tmp_path.resolve()
        assert s.outputs_dir == tmp_path.resolve() / "outputs"


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_auth_disabled_by_default(self):
        from src.config import settings
        assert settings.api_key == ""

    def test_verify_api_key_passes_when_disabled(self):
        from src.api.auth import verify_api_key
        # No exception when auth is off (api_key == "")
        asyncio.run(verify_api_key(x_api_key=None))

    def test_verify_api_key_raises_for_wrong_key(self, monkeypatch):
        import src.api.auth as auth_module
        from src.config import Settings
        monkeypatch.setenv("PRICING_API_KEY", "secret")
        patched = Settings()
        monkeypatch.setattr(auth_module, "settings", patched)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(auth_module.verify_api_key(x_api_key="wrong"))
        assert exc_info.value.status_code == 401

    def test_verify_api_key_passes_for_correct_key(self, monkeypatch):
        import src.api.auth as auth_module
        from src.config import Settings
        monkeypatch.setenv("PRICING_API_KEY", "secret")
        patched = Settings()
        monkeypatch.setattr(auth_module, "settings", patched)
        # Correct key — should not raise
        asyncio.run(auth_module.verify_api_key(x_api_key="secret"))


# ---------------------------------------------------------------------------
# Multi-key rotation support (TIER 3.1)
# ---------------------------------------------------------------------------

class TestApiKeyRotation:
    def test_valid_keys_empty_by_default(self):
        from src.config import Settings
        assert Settings().valid_keys == frozenset()

    def test_valid_keys_from_single_api_key(self):
        from src.config import Settings
        assert Settings(api_key="secret1").valid_keys == frozenset({"secret1"})

    def test_valid_keys_from_api_keys_csv(self):
        from src.config import Settings
        assert Settings(api_keys="key1,key2,key3").valid_keys == frozenset({"key1", "key2", "key3"})

    def test_valid_keys_strips_whitespace(self):
        from src.config import Settings
        assert Settings(api_keys="key1 , key2").valid_keys == frozenset({"key1", "key2"})

    def test_valid_keys_skips_empty_entries(self):
        from src.config import Settings
        assert Settings(api_keys="key1,,key2").valid_keys == frozenset({"key1", "key2"})

    def test_valid_keys_is_union_of_both_fields(self):
        from src.config import Settings
        s = Settings(api_key="primary", api_keys="secondary,tertiary")
        assert s.valid_keys == frozenset({"primary", "secondary", "tertiary"})

    def test_api_keys_env_override(self, monkeypatch):
        monkeypatch.setenv("PRICING_API_KEYS", "a,b,c")
        from src.config import Settings
        assert Settings().valid_keys == frozenset({"a", "b", "c"})

    def test_verify_passes_for_any_key_in_set(self, monkeypatch):
        import src.api.auth as auth_module
        from src.config import Settings
        patched = Settings(api_keys="key1,key2")
        monkeypatch.setattr(auth_module, "settings", patched)
        asyncio.run(auth_module.verify_api_key(x_api_key="key1"))
        asyncio.run(auth_module.verify_api_key(x_api_key="key2"))

    def test_verify_rejects_unknown_key_when_multi_key_configured(self, monkeypatch):
        import src.api.auth as auth_module
        from src.config import Settings
        patched = Settings(api_keys="key1,key2")
        monkeypatch.setattr(auth_module, "settings", patched)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(auth_module.verify_api_key(x_api_key="unknown"))
        assert exc_info.value.status_code == 401

    def test_verify_rejects_none_when_multi_key_configured(self, monkeypatch):
        import src.api.auth as auth_module
        from src.config import Settings
        patched = Settings(api_keys="key1,key2")
        monkeypatch.setattr(auth_module, "settings", patched)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(auth_module.verify_api_key(x_api_key=None))
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Metrics module import (ensures coverage gate sees src/api/metrics.py)
# ---------------------------------------------------------------------------

class TestMetricsImport:
    def test_metrics_collectors_importable(self):
        from src.api.metrics import (
            CACHE_HITS,
            CACHE_MISSES,
            REQUEST_COUNT,
            REQUEST_LATENCY,
        )
        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None
        assert CACHE_HITS is not None
        assert CACHE_MISSES is not None


# ---------------------------------------------------------------------------
# Tier 4 settings: log_format, allowed_origins, simulate_rate_limit
# ---------------------------------------------------------------------------

class TestTier4Settings:
    def test_log_format_default_is_text(self):
        from src.config import settings
        assert settings.log_format == "text"

    def test_log_format_json_accepted(self):
        from src.config import Settings
        s = Settings(log_format="json")
        assert s.log_format == "json"

    def test_log_format_invalid_rejected(self):
        from pydantic import ValidationError

        from src.config import Settings
        with pytest.raises(ValidationError):
            Settings(log_format="yaml")

    def test_allowed_origins_default(self):
        from src.config import settings
        assert settings.allowed_origins == "*"

    def test_allowed_origins_env_override(self, monkeypatch):
        monkeypatch.setenv("PRICING_ALLOWED_ORIGINS", "https://a.com,https://b.com")
        from src.config import Settings
        s = Settings()
        # stored as-is; main.py splits on comma when building CORSMiddleware
        assert s.allowed_origins == "https://a.com,https://b.com"

    def test_allowed_origins_parses_to_list(self, monkeypatch):
        monkeypatch.setenv("PRICING_ALLOWED_ORIGINS", "https://a.com,https://b.com")
        from src.config import Settings
        s = Settings()
        origins = [o.strip() for o in s.allowed_origins.split(",") if o.strip()]
        assert origins == ["https://a.com", "https://b.com"]

    def test_simulate_rate_limit_default(self):
        from src.config import settings
        assert settings.simulate_rate_limit == 30

    def test_simulate_rate_limit_override(self, monkeypatch):
        monkeypatch.setenv("PRICING_SIMULATE_RATE_LIMIT", "100")
        from src.config import Settings
        s = Settings()
        assert s.simulate_rate_limit == 100


# ---------------------------------------------------------------------------
# JsonFormatter — structured log output
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    def _make_record(self, msg: str = "hello", level: int = logging.INFO, **extra):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_produces_valid_json(self):
        from src.api.logging import JsonFormatter
        formatter = JsonFormatter()
        output = formatter.format(self._make_record("startup"))
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "startup"
        assert "timestamp" in parsed
        assert "logger" in parsed

    def test_includes_extra_access_fields(self):
        from src.api.logging import JsonFormatter
        formatter = JsonFormatter()
        record = self._make_record(
            "req",
            request_id="abc12345",
            method="GET",
            path="/health",
            status=200,
            duration_ms=1.5,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["request_id"] == "abc12345"
        assert parsed["method"] == "GET"
        assert parsed["path"] == "/health"
        assert parsed["status"] == 200
        assert parsed["duration_ms"] == pytest.approx(1.5)

    def test_extra_fields_absent_when_not_set(self):
        from src.api.logging import JsonFormatter
        formatter = JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record("startup")))
        for field in ("request_id", "method", "path", "status", "duration_ms"):
            assert field not in parsed

    def test_warning_level_captured(self):
        from src.api.logging import JsonFormatter
        formatter = JsonFormatter()
        record = self._make_record("oops", level=logging.WARNING)
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"
