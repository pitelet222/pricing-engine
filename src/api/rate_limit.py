"""
rate_limit.py — Sliding-window rate limiter for POST /simulate.

Limits each client IP to PRICING_SIMULATE_RATE_LIMIT requests per 60-second
window. Returns 429 Too Many Requests with a Retry-After header when the
limit is exceeded.

Two backends are supported:

* In-process (default): a ``dict`` of deques. Each worker process tracks its
  own window — fine for local dev and single-instance deployments, but not
  coordinated across replicas.
* Redis-backed (set PRICING_REDIS_URL): a sorted-set sliding window shared by
  all workers/replicas, giving consistent limits across a multi-process
  deployment.

If Redis is configured but unreachable, the limiter logs a warning and falls
back to the in-process window rather than blocking all traffic — availability
matters more than strict enforcement for a non-critical guard like this.
"""

from __future__ import annotations

import collections
import logging
import time
import uuid

import redis
from fastapi import HTTPException, Request

from src.config import settings

_log = logging.getLogger(__name__)

# {client_ip: deque of request timestamps (monotonic seconds)} — in-process fallback.
_windows: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque())

_WINDOW_SECS = 60

_redis_client: redis.Redis | None = None
_redis_broken = False


def _get_redis_client() -> redis.Redis | None:
    """Lazily create and cache a Redis client, or return None if unavailable."""
    global _redis_client, _redis_broken

    if not settings.redis_url or _redis_broken:
        return None

    if _redis_client is None:
        _redis_client = redis.Redis.from_url(settings.redis_url)

    return _redis_client


def _check_redis(client: redis.Redis, ip: str) -> bool:
    """
    Record a request for `ip` and report whether it exceeds the limit, using a
    Redis sorted-set sliding window shared across all processes.

    Returns True if the request is within the limit (and has been recorded),
    False if the limit is exceeded (the request is not recorded as a hit).
    """
    key = f"ratelimit:simulate:{ip}"
    now = time.time()
    cutoff = now - _WINDOW_SECS

    pipe = client.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff)
    pipe.zcard(key)
    _, count = pipe.execute()

    if count >= settings.simulate_rate_limit:
        return False

    member = f"{now:.6f}:{uuid.uuid4().hex}"
    pipe = client.pipeline()
    pipe.zadd(key, {member: now})
    pipe.expire(key, _WINDOW_SECS)
    pipe.execute()
    return True


def _check_in_process(ip: str) -> bool:
    """Record a request for `ip` and report whether it's within the limit."""
    now = time.monotonic()
    window = _windows[ip]

    while window and window[0] < now - _WINDOW_SECS:
        window.popleft()

    if len(window) >= settings.simulate_rate_limit:
        return False

    window.append(now)
    return True


def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency: enforce per-IP sliding-window rate limit.

    Raises
    ------
    HTTPException(429)
        When the client has exceeded PRICING_SIMULATE_RATE_LIMIT requests
        within the last 60 seconds. Includes a Retry-After header.
    """
    global _redis_broken

    ip: str = request.client.host if request.client else "unknown"

    client = _get_redis_client()
    if client is not None:
        try:
            allowed = _check_redis(client, ip)
        except Exception:
            _log.warning(
                "Redis rate limiter unreachable — falling back to in-process limiting",
                exc_info=True,
            )
            _redis_broken = True
            allowed = _check_in_process(ip)
    else:
        allowed = _check_in_process(ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {settings.simulate_rate_limit} requests "
                f"per minute per IP on POST /simulate. Try again later."
            ),
            headers={"Retry-After": str(_WINDOW_SECS)},
        )
