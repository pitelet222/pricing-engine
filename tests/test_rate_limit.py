"""
Unit tests for the in-process sliding-window rate limiter.

No model artifacts required — always runs in CI.
Uses monkeypatching to reduce the limit so tests stay fast (no need to loop
30 times for every assertion).
"""
from __future__ import annotations

from unittest.mock import MagicMock

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
