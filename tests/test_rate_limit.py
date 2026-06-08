"""
Unit tests for the sliding-window rate limiter (in-process and Redis-backed).

No model artifacts required — always runs in CI.
Uses monkeypatching to reduce the limit so tests stay fast (no need to loop
30 times for every assertion). The Redis-backed path is exercised against
fakeredis so the suite never needs a live server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import pytest
from fastapi import HTTPException

import src.api.rate_limit as rl


def _mock_request(ip: str = "1.2.3.4") -> MagicMock:
    req = MagicMock()
    req.client.host = ip
    return req


class TestRateLimit:
    def setup_method(self):
        """Isolate each test by clearing the shared window state."""
        rl._windows.clear()

    def test_allows_requests_under_limit(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 5)
        req = _mock_request("10.0.0.1")
        for _ in range(5):
            rl.check_rate_limit(req)  # must not raise

    def test_blocks_request_after_limit_exceeded(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 3)
        req = _mock_request("10.0.0.2")
        for _ in range(3):
            rl.check_rate_limit(req)
        with pytest.raises(HTTPException) as exc_info:
            rl.check_rate_limit(req)
        assert exc_info.value.status_code == 429

    def test_retry_after_header_present(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 2)
        req = _mock_request("10.0.0.3")
        for _ in range(2):
            rl.check_rate_limit(req)
        with pytest.raises(HTTPException) as exc_info:
            rl.check_rate_limit(req)
        assert "Retry-After" in exc_info.value.headers

    def test_retry_after_value_is_60(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 1)
        req = _mock_request("10.0.0.4")
        rl.check_rate_limit(req)
        with pytest.raises(HTTPException) as exc_info:
            rl.check_rate_limit(req)
        assert exc_info.value.headers["Retry-After"] == "60"

    def test_different_ips_tracked_independently(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 2)
        req_a = _mock_request("192.168.0.1")
        req_b = _mock_request("192.168.0.2")
        for _ in range(2):
            rl.check_rate_limit(req_a)
        # req_b has its own empty window — must not raise
        rl.check_rate_limit(req_b)

    def test_unknown_client_ip_handled(self, monkeypatch):
        """request.client = None falls back to 'unknown' key."""
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 2)
        req = MagicMock()
        req.client = None
        for _ in range(2):
            rl.check_rate_limit(req)
        with pytest.raises(HTTPException):
            rl.check_rate_limit(req)


class TestRateLimitRedis:
    """Exercises the Redis-backed sliding window via fakeredis."""

    def setup_method(self):
        rl._windows.clear()
        rl._redis_client = None
        rl._redis_broken = False

    def teardown_method(self):
        rl._redis_client = None
        rl._redis_broken = False

    def _use_fake_redis(self, monkeypatch):
        fake = fakeredis.FakeRedis()
        monkeypatch.setattr(rl.settings, "redis_url", "redis://fake:6379/0")
        monkeypatch.setattr(rl, "_get_redis_client", lambda: fake)
        return fake

    def test_allows_requests_under_limit(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 5)
        self._use_fake_redis(monkeypatch)
        req = _mock_request("10.1.0.1")
        for _ in range(5):
            rl.check_rate_limit(req)  # must not raise

    def test_blocks_request_after_limit_exceeded(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 3)
        self._use_fake_redis(monkeypatch)
        req = _mock_request("10.1.0.2")
        for _ in range(3):
            rl.check_rate_limit(req)
        with pytest.raises(HTTPException) as exc_info:
            rl.check_rate_limit(req)
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["Retry-After"] == "60"

    def test_state_shared_across_instances(self, monkeypatch):
        """Two independent dependency calls against the same backend share state."""
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 2)
        fake = fakeredis.FakeRedis()
        monkeypatch.setattr(rl.settings, "redis_url", "redis://fake:6379/0")
        monkeypatch.setattr(rl, "_get_redis_client", lambda: fake)

        req = _mock_request("10.1.0.3")
        rl.check_rate_limit(req)
        rl.check_rate_limit(req)
        with pytest.raises(HTTPException):
            # A "fresh" lookup against the same fake server must still see
            # the two hits recorded above — proving state isn't per-process.
            rl.check_rate_limit(req)

    def test_falls_back_to_in_process_when_redis_unreachable(self, monkeypatch):
        """Fail-open: a broken Redis client must not block traffic outright."""
        monkeypatch.setattr(rl.settings, "simulate_rate_limit", 2)
        monkeypatch.setattr(rl.settings, "redis_url", "redis://fake:6379/0")

        broken_client = MagicMock()
        broken_client.pipeline.side_effect = ConnectionError("boom")
        monkeypatch.setattr(rl, "_get_redis_client", lambda: broken_client)

        req = _mock_request("10.1.0.4")
        rl.check_rate_limit(req)  # Redis raises, falls back, recorded in-process
        rl.check_rate_limit(req)
        with pytest.raises(HTTPException) as exc_info:
            rl.check_rate_limit(req)
        assert exc_info.value.status_code == 429
        assert rl._redis_broken is True

    def test_get_redis_client_returns_none_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "redis_url", "")
        assert rl._get_redis_client() is None

    def test_get_redis_client_returns_none_when_marked_broken(self, monkeypatch):
        monkeypatch.setattr(rl.settings, "redis_url", "redis://fake:6379/0")
        rl._redis_broken = True
        assert rl._get_redis_client() is None
