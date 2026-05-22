# =============================================================================
# Avocado Pricing Engine — Developer Makefile
#
# Requires GNU make and a POSIX shell (bash/sh).
# On Windows without Git Bash / WSL, use the Python alternative:
#   python scripts/bootstrap.py <target>
# =============================================================================

.DEFAULT_GOAL := help

PYTHON := python
PIP    := $(PYTHON) -m pip

.PHONY: help setup setup-full test test-integration test-all lint lint-types api dashboard notebooks clean

help: ## Show this help message
	@printf "\n\033[1mAvocado Pricing Engine\033[0m — available targets:\n\n"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*## / \
		{printf "  \033[36m%-20s\033[0m%s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup: ## Install dev dependencies (pytest, ruff, coverage)
	$(PIP) install -r requirements-dev.txt

setup-full: ## Install all dependencies including the full ML stack
	$(PIP) install -r requirements.txt -r requirements-dev.txt

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: ## Run all CI-safe unit tests (no model artefacts required)
	$(PYTHON) -m pytest \
		tests/test_schemas.py tests/test_config.py tests/test_rate_limit.py \
		tests/test_preprocessing.py tests/test_features.py tests/test_metrics.py \
		tests/test_forecaster.py tests/test_pricer.py tests/test_uncertainty.py \
		tests/test_explainability.py tests/test_charts.py tests/test_loader_unit.py \
		tests/test_manifest.py \
		-m "not integration" -v

test-integration: ## Run toy-DataStore integration tests (requires lightgbm + scikit-learn)
	$(PYTHON) -m pytest tests/test_integration.py -v

test-all: ## Run every test file, including artefact-gated and integration tests
	$(PYTHON) -m pytest -v

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Check code style with ruff
	$(PYTHON) -m ruff check src/ tests/ scripts/

lint-types: ## Run mypy static type checker on src/
	$(PYTHON) -m mypy src/

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

api: ## Start the FastAPI server with hot-reload (http://localhost:8000)
	$(PYTHON) -m uvicorn src.api.main:app --reload

dashboard: ## Start the Streamlit dashboard (http://localhost:8501)
	$(PYTHON) -m streamlit run src/dashboard/app.py

# ---------------------------------------------------------------------------
# Data pipeline — runs notebooks in order to regenerate data/outputs/
# WARNING: takes 30–60 min due to neural-forecast training in notebook 03.
# ---------------------------------------------------------------------------

notebooks: ## Execute notebooks 01–06 in order (generates data/outputs/)
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_forecasting.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/04_pricing.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/05_uncertainty.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/06_explainability.ipynb

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove Python caches, pytest artefacts, and coverage reports
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage"     -delete 2>/dev/null || true
