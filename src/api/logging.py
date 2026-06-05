"""
logging.py — JSON log formatter for structured log ingestion.

Activated via PRICING_LOG_FORMAT=json. Each log record is emitted as a
single JSON line, parseable by Logstash, Fluentd, Grafana Loki, etc.

Structured fields attached by AccessLogMiddleware (request_id, method,
path, status, duration_ms) are promoted to top-level JSON keys so log
queries can filter by them directly without string parsing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Emit every log record as a single JSON line on stdout."""

    _EXTRA_FIELDS = ("request_id", "method", "path", "status", "duration_ms")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self._EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
