"""
config.py — Centralised application settings.

All values have sensible defaults for local development.  Any setting can be
overridden at deploy-time via an environment variable prefixed with PRICING_:

    PRICING_LOG_LEVEL=DEBUG
    PRICING_DATA_ROOT=/mnt/shared/data

Docker Compose passes these through the ``environment:`` block.
A .env file in the project root is also picked up automatically.
"""

from __future__ import annotations

import pathlib
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # pricing-engine/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRICING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    data_root: pathlib.Path = _PROJECT_ROOT / "data"

    @field_validator("data_root", mode="before")
    @classmethod
    def _resolve(cls, v: str | pathlib.Path) -> pathlib.Path:
        return pathlib.Path(v).resolve()

    @property
    def outputs_dir(self) -> pathlib.Path:
        return self.data_root / "outputs"

    @property
    def processed_dir(self) -> pathlib.Path:
        return self.data_root / "processed"

    # --------------------------------------------------------------- app meta
    api_title: str = "Avocado Pricing Engine"
    api_version: str = "0.1.0"
    api_description: str = (
        "Dynamic pricing recommendations, demand forecasts, uncertainty "
        "quantification, and SHAP explainability for 86 avocado market series."
    )

    # --------------------------------------------------------------- security
    # Auth is disabled when both fields are empty (local dev / test default).
    #
    # Single-key mode:
    #   PRICING_API_KEY=your-secret
    #
    # Multi-key rotation (old key stays valid while clients migrate):
    #   PRICING_API_KEYS=new-secret,old-secret
    #
    # Setting both produces the union — all listed keys are accepted.
    api_key: str = ""
    api_keys: str = ""  # comma-separated; merged with api_key in valid_keys

    @property
    def valid_keys(self) -> frozenset[str]:
        """All accepted API keys. Empty frozenset means auth is disabled."""
        keys: set[str] = set()
        if self.api_key:
            keys.add(self.api_key)
        if self.api_keys:
            keys.update(k.strip() for k in self.api_keys.split(",") if k.strip())
        return frozenset(keys)

    # -------------------------------------------------------------- observability
    log_level: str = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    # "text" = human-readable (development); "json" = structured lines (production/ELK)
    log_format: Literal["text", "json"] = "text"

    # ------------------------------------------------------------------ CORS
    # Comma-separated string of allowed origins.
    #   PRICING_ALLOWED_ORIGINS=https://app.example.com,https://staging.example.com
    # Default "*" is safe for local dev; lock down in production.
    # Stored as str so the env var can be set naturally without JSON escaping.
    allowed_origins: str = "*"

    # ------------------------------------------------------- rate limiting
    # Maximum POST /simulate requests per IP per 60-second window.
    simulate_rate_limit: int = 30

    # Redis URL for distributed rate-limit state, e.g. redis://localhost:6379/0.
    # Empty string (default) disables Redis: each worker falls back to an
    # in-process sliding window — correct for local dev and single-instance
    # deployments, but workers won't share state across processes/replicas.
    redis_url: str = ""


settings = Settings()
