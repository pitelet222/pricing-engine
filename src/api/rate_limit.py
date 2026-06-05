"""
rate_limit.py — In-process sliding-window rate limiter for POST /simulate.

Limits each client IP to PRICING_SIMULATE_RATE_LIMIT requests per 60-second
window. Designed for single-process deployments; each worker maintains its own
window. For multi-process or distributed deployments, replace _windows with a
Redis-backed equivalent.

Returns 429 Too Many Requests with a Retry-After header when the limit is
exceeded.
"""

from __future__ import annotations

import collections
import time

from fastapi import HTTPException, Request

from src.config import settings

# {client_ip: deque of request timestamps (monotonic seconds)}
_windows: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque())

_WINDOW_SECS = 60


def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency: enforce per-IP sliding-window rate limit.

    Raises
    ------
    HTTPException(429)
        When the client has exceeded PRICING_SIMULATE_RATE_LIMIT requests
        within the last 60 seconds. Includes a Retry-After header.
    """
    ip: str = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _windows[ip]

    # Evict timestamps that have slid out of the window.
    while window and window[0] < now - _WINDOW_SECS:
        window.popleft()

    if len(window) >= settings.simulate_rate_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {settings.simulate_rate_limit} requests "
                f"per minute per IP on POST /simulate. Try again later."
            ),
            headers={"Retry-After": str(_WINDOW_SECS)},
        )

    window.append(now)
