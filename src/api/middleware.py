"""
middleware.py — ASGI middleware for request tracing and observability.

AccessLogMiddleware does three things per request:

1. X-Request-ID tracing
   Generates a short UUID hex prefix (8 chars) for every request, stores it in
   request.state.request_id, and returns it in the X-Request-ID response header.
   Ties all log lines from a single request together — essential for debugging
   when multiple requests are in-flight simultaneously.

2. Structured access logging
   Logs method, path, status code, and wall-clock duration in milliseconds:
       2024-01-07T12:34:56 [INFO] api.access: [a1b2c3d4] GET /forecast/Albany_conventional 200 4.2ms

3. Prometheus instrumentation
   Increments api_requests_total (by method, endpoint template, status code)
   and records api_request_duration_seconds (by endpoint template).
   Uses the FastAPI route template (e.g. /forecast/{unique_id}) rather than
   the raw path to avoid label cardinality explosion across 86 series.
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.api.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger("api.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id

        t0 = time.perf_counter()
        response = await call_next(request)
        duration_s = time.perf_counter() - t0

        # Use the route template for Prometheus labels to avoid cardinality explosion.
        route = request.scope.get("route")
        endpoint_label = route.path if route else request.url.path

        logger.info(
            "[%s] %s %s %d %.1fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_s * 1000,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_s * 1000, 1),
            },
        )

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint_label,
            status_code=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint_label).observe(duration_s)

        response.headers["X-Request-ID"] = request_id
        return response
