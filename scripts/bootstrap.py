#!/usr/bin/env python3
"""
bootstrap.py — Cross-platform build helper (Windows-friendly Makefile alternative).

Mirrors every target in the project Makefile so the same workflow commands
work on all operating systems without requiring GNU make.

Usage:
    python scripts/bootstrap.py <target>
    python scripts/bootstrap.py --help

Targets:
    setup            Install dev dependencies (pytest, ruff, coverage)
    setup-full       Install all dependencies including the full ML stack
    test             Run CI-safe unit tests (no model artefacts required)
    test-integration Run toy-DataStore integration tests (requires lightgbm)
    test-all         Run every test file
    lint             Check code style with ruff
    api              Start the FastAPI server with hot-reload
    dashboard        Start the Streamlit dashboard
    notebooks        Execute notebooks 01–06 in order
    clean            Remove Python caches and coverage reports
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PY = sys.executable


def _run(*args: str) -> None:
    """Run a subprocess command from the project root; exit on failure."""
    result = subprocess.run(list(args), cwd=ROOT)
    if result.returncode:
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Target implementations
# ---------------------------------------------------------------------------

def _setup() -> None:
    _run(_PY, "-m", "pip", "install", "-r", "requirements-dev.txt")


def _setup_full() -> None:
    _run(
        _PY, "-m", "pip", "install",
        "-r", "requirements.txt",
        "-r", "requirements-dev.txt",
    )


def _test() -> None:
    _run(
        _PY, "-m", "pytest",
        "tests/test_schemas.py",
        "tests/test_config.py",
        "tests/test_rate_limit.py",
        "tests/test_preprocessing.py",
        "tests/test_features.py",
        "tests/test_metrics.py",
        "tests/test_forecaster.py",
        "tests/test_pricer.py",
        "tests/test_uncertainty.py",
        "tests/test_explainability.py",
        "tests/test_charts.py",
        "tests/test_loader_unit.py",
        "-m", "not integration",
        "-v",
    )


def _test_integration() -> None:
    _run(_PY, "-m", "pytest", "tests/test_integration.py", "-v")


def _test_all() -> None:
    _run(_PY, "-m", "pytest", "-v")


def _lint() -> None:
    _run(_PY, "-m", "ruff", "check", "src/", "tests/", "scripts/")


def _api() -> None:
    _run(_PY, "-m", "uvicorn", "src.api.main:app", "--reload")


def _dashboard() -> None:
    _run(_PY, "-m", "streamlit", "run", "src/dashboard/app.py")


def _notebooks() -> None:
    for nb in (
        "notebooks/01_eda.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_forecasting.ipynb",
        "notebooks/04_pricing.ipynb",
        "notebooks/05_uncertainty.ipynb",
        "notebooks/06_explainability.ipynb",
    ):
        _run(
            _PY, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace", nb,
        )


def _clean() -> None:
    for pattern in ("**/__pycache__", "**/.pytest_cache", "**/htmlcov"):
        for p in ROOT.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    for pattern in ("**/*.pyc", "**/.coverage"):
        for p in ROOT.glob(pattern):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    print("Clean complete.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TARGETS: dict[str, tuple[object, str]] = {
    "setup":            (_setup,            "Install dev dependencies"),
    "setup-full":       (_setup_full,       "Install all dependencies including ML stack"),
    "test":             (_test,             "Run CI-safe unit tests"),
    "test-integration": (_test_integration, "Run toy-DataStore integration tests"),
    "test-all":         (_test_all,         "Run all tests"),
    "lint":             (_lint,             "Lint with ruff"),
    "api":              (_api,              "Start FastAPI server (http://localhost:8000)"),
    "dashboard":        (_dashboard,        "Start Streamlit dashboard (http://localhost:8501)"),
    "notebooks":        (_notebooks,        "Execute notebooks 01–06 (30–60 min)"),
    "clean":            (_clean,            "Remove Python caches and coverage reports"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-platform build helper — mirrors the project Makefile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Targets:\n" + "\n".join(
            f"  {name:<20}{desc}"
            for name, (_, desc) in _TARGETS.items()
        ),
    )
    parser.add_argument(
        "target",
        choices=list(_TARGETS),
        metavar="target",
        help=f"Build target ({', '.join(_TARGETS)})",
    )
    args = parser.parse_args()
    fn, _ = _TARGETS[args.target]
    fn()  # type: ignore[operator]


if __name__ == "__main__":
    main()
