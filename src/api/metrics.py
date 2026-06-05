"""
metrics.py — Prometheus metric collectors.

All counters and histograms are module-level singletons (created once at import
time). They are updated by AccessLogMiddleware (request count, latency) and by
the route cache logic (cache hits/misses).

Exposed via GET /metrics in text/plain Prometheus exposition format.
Consume with any Prometheus scraper; visualise with Grafana.

Cardinality note
----------------
The `endpoint` label uses the FastAPI route template (e.g. `/forecast/{unique_id}`)
rather than the raw path, so 86 series don't create 86 distinct time-series.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Request metrics — updated by AccessLogMiddleware
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests handled",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ---------------------------------------------------------------------------
# Cache metrics — updated by route handlers
# ---------------------------------------------------------------------------

CACHE_HITS = Counter(
    "api_cache_hits_total",
    "Number of times a cached response was served",
    ["endpoint"],
)

CACHE_MISSES = Counter(
    "api_cache_misses_total",
    "Number of times a response had to be computed (cache miss)",
    ["endpoint"],
)
