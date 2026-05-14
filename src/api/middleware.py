"""
middleware.py — ASGI middleware for structured request/response logging.

Every handled request is logged at INFO level with method, path, HTTP status,
and wall-clock duration in milliseconds:

    2024-01-07T12:34:56 [INFO] api.access: GET /forecast/Albany_conventional 200 4.2ms

Unhandled exceptions bubble up to FastAPI's default 500 handler; the duration
is still logged so slow-failing requests are visible.
"""
from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        return response
