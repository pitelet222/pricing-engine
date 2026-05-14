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

    # -------------------------------------------------------------- observability
    log_level: str = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


settings = Settings()
