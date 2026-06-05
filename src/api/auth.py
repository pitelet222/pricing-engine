"""
auth.py — API key authentication dependency.

Applied as a global FastAPI dependency so every business endpoint is
protected by default. Auth is silently disabled when no keys are configured —
this is the default for local development and tests.

Usage
-----
Single-key mode:
    PRICING_API_KEY=your-secret uvicorn src.api.main:app ...

Multi-key rotation (keep old key active while clients migrate to new one):
    PRICING_API_KEYS=new-secret,old-secret uvicorn src.api.main:app ...

Or in docker-compose.yml:
    environment:
      - PRICING_API_KEY=your-secret
      # OR for rotation:
      # - PRICING_API_KEYS=new-secret,old-secret

Clients pass the key in the X-API-Key header:
    curl -H "X-API-Key: your-secret" http://localhost:8000/series
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from src.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency that validates the X-API-Key header.

    - If no keys are configured (PRICING_API_KEY and PRICING_API_KEYS both
      empty): always passes — auth disabled, for local dev and tests.
    - If any keys are configured: the header must be present and must match
      one of the accepted keys exactly.

    Raises
    ------
    HTTPException(401)
        When auth is enabled and the key is missing or not in the valid set.
    """
    valid_keys = settings.valid_keys
    if not valid_keys:
        return  # auth disabled — all requests pass

    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid or missing API key. Provide your key in the X-API-Key request header."
            ),
        )
