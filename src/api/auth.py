"""
auth.py — API key authentication dependency.

Applied as a global FastAPI dependency so every business endpoint is
protected by default. Auth is silently disabled when PRICING_API_KEY is
not set (empty string) — this is the default for local development and tests.

Usage
-----
To enable auth in production / Docker:
    PRICING_API_KEY=your-secret-here uvicorn src.api.main:app ...

Or in docker-compose.yml:
    environment:
      - PRICING_API_KEY=your-secret-here

Clients pass the key in the X-API-Key header:
    curl -H "X-API-Key: your-secret-here" http://localhost:8000/series
"""
from __future__ import annotations

from fastapi import Header, HTTPException

from src.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency that validates the X-API-Key header.

    - If PRICING_API_KEY is not configured (empty string): always passes.
    - If configured: the header must be present and must match exactly.

    Raises
    ------
    HTTPException(401)
        When auth is enabled and the key is missing or incorrect.
    """
    if not settings.api_key:
        return  # auth disabled — all requests pass

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid or missing API key. "
                "Provide your key in the X-API-Key request header."
            ),
        )
