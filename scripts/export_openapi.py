"""
export_openapi.py — Dump the FastAPI OpenAPI spec to docs/openapi.json.

Usage:
    python scripts/export_openapi.py

The generated file must be committed alongside the source. CI verifies it stays
current by re-generating and diffing:
    python scripts/export_openapi.py && git diff --exit-code docs/openapi.json

Notes
-----
Importing src.api.main triggers logging setup but NOT the lifespan, so no
DataStore loading occurs. app.openapi() introspects route definitions only.
"""

from __future__ import annotations

import json
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.main import app  # noqa: E402 — must come after sys.path insert

_DOCS_DIR = _PROJECT_ROOT / "docs"
_DOCS_DIR.mkdir(exist_ok=True)
_OUT = _DOCS_DIR / "openapi.json"

spec = app.openapi()
_OUT.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print(f"Written {len(spec['paths'])} paths -> {_OUT}")
