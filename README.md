<div align="center">

# 🥑 Avocado Pricing Engine

### *Stop guessing. Start pricing with data.*

A full-stack machine learning system that forecasts market prices, recommends revenue-optimal prices, and explains every decision — built for produce distributors, retailers, and anyone competing in a price-sensitive market.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-green)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/pitelet222/pricing-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/pitelet222/pricing-engine/actions/workflows/ci.yml)
[![CD](https://github.com/pitelet222/pricing-engine/actions/workflows/cd.yml/badge.svg)](https://github.com/pitelet222/pricing-engine/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/pitelet222/pricing-engine/graph/badge.svg)](https://codecov.io/gh/pitelet222/pricing-engine)

[**The Business Problem**](#-the-business-problem) · [**What It Does**](#-what-it-does) · [**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**API**](#-api-reference) · [**Results**](#-results)

---

</div>

## 🏪 The Business Problem

Every retailer faces the same dilemma: **set the price too high and lose customers; set it too low and leave money on the table.**

For perishable goods like fresh produce, this is even harder:
- Prices fluctuate week to week across dozens of regions
- Demand responds differently to price changes in different markets
- A price that works in Los Angeles might fail in Denver
- There is no easy way to know *why* a pricing decision is good or bad

Most businesses rely on intuition, competitor price-matching, or simple rules of thumb. The result is systematic underpricing or overpricing — often without ever knowing it.

**This engine changes that.** It learns the price-demand relationship from historical data, forecasts where prices are heading, and recommends the specific price for each market that maximises revenue — with a plain-English explanation of every recommendation.

---

## ✅ What It Does

| Capability | What it means for your business |
|---|---|
| **12-week price forecast** | Know where market prices are heading before your competitors do |
| **Revenue-optimal pricing** | For each region and product type, find the exact price that maximises weekly revenue |
| **Risk-aware strategies** | Choose between Conservative, Balanced, or Aggressive pricing based on how much uncertainty you can tolerate |
| **Explainable recommendations** | Every price recommendation comes with a plain-English reason: *"Demand in this market is inelastic — raising price by 8% increases revenue by $1,200/week"* |
| **Live simulation** | Test any price before committing: *"If I set price to $1.80 in Chicago, what does the model predict for volume?"* |
| **Uncertainty quantification** | 90% prediction intervals so you know how confident the model is — and which markets are high-risk |

---

## 🚀 Quick Start

### Option A — Docker (recommended, zero setup)

```bash
git clone https://github.com/pitelet222/pricing-engine.git
cd pricing-engine

# Run all notebooks to generate model artifacts (one-time)
# Then launch API + dashboard together:
docker-compose up
```

- Dashboard → http://localhost:8501
- API docs → http://localhost:8000/docs

### Option B — Local Python

```bash
# 1. Clone and install
git clone https://github.com/pitelet222/pricing-engine.git
cd pricing-engine
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Generate model artifacts (run notebooks 01 → 06 in order)
make notebooks        # or: jupyter notebook, then run 01 → 06 manually

# 3. Start the API
make api              # or: uvicorn src.api.main:app --reload

# 4. Start the dashboard (separate terminal)
make dashboard        # or: streamlit run src/dashboard/app.py
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                        │
│                                                             │
│  avocado.csv → [01 EDA] → [02 Features] → [03 Forecast]     │
│                                    ↓              ↓         │
│                             [04 Pricing] → [05 Uncertainty] │
│                                    ↓                        │
│                            [06 Explainability]              │
│                                    ↓                        |
│                          data/outputs/  ←─── artefacts ──── │
└───────────────────────────────┬─────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
    ┌─────────▼───────-┐ ┌──────▼──────┐  ┌───────▼───────┐
    │   FastAPI        │ │  Streamlit  │  │  Jupyter      │
    │  src/api/        │ │  Dashboard  │  │  Notebooks    │
    │                  │ │ src/dash../ │  │               │
    │ /v1/series       │ │             │  │  Analysis &   │
    │ /v1/forecast/{id}│ │  Forecast   │  │  Research     │
    │ /v1/recommend/{id}││  Pricing    │  │               │
    │ /v1/explain/{id} │ │  Explain    │  └───────────────┘
    │ /v1/simulate     │ │  Simulate   │
    └──────────────────┘ └─────────────┘
```

### Forecasting: 7-Model Ensemble

```
┌─────────────────────────────────────────────────────────┐
│  Statistical (MSTL decomposition)                       │
│    MSTL + AutoARIMA  ──┐                                │
│    MSTL + AutoETS    ──┤──► Ensemble_weighted           │
│    MSTL + AutoTheta  ──┤    (inverse-MAE weights)       │
│                        │                               │
│  Neural                │                               │
│    NHITS             ──┤                                │
│    NBEATSx           ──┤                                │
│    DLinear           ──┘                                │
│                                                         │
│  Baseline                                               │
│    SeasonalNaive  (benchmark)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Results

Evaluated on a 12-week temporal holdout across 86 market series (43 US regions × 2 product types):

| Model | MAE | RMSE | SMAPE |
|---|---|---|---|
| **Ensemble_weighted** | **0.152** | **0.198** | **11.6%** |
| MSTL_ETS | 0.168 | 0.214 | 12.9% |
| MSTL_ARIMA | 0.171 | 0.218 | 13.1% |
| NHITS | 0.183 | 0.231 | 14.0% |
| SeasonalNaive | 0.221 | 0.278 | 17.2% |

**Pricing recommendations** (86 series, balanced strategy):
- Median revenue uplift opportunity: **+21%**
- Series with >5% uplift identified: **81 / 86**
- Conservative strategy median uplift: **+17%**

> ⚠️ Revenue uplift figures are model estimates on the training distribution. A/B testing is recommended before full deployment.

---

## 📁 Project Structure

```
pricing-engine/
├── notebooks/                    # Full analysis pipeline (run in order)
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_forecasting.ipynb      # 7-model ensemble price forecasting
│   ├── 04_pricing.ipynb          # LightGBM demand model + revenue optimisation
│   ├── 05_uncertainty.ipynb      # Conformal PIs + quantile strategies
│   └── 06_explainability.ipynb   # SHAP attribution + business narratives
│
├── src/
│   ├── api/                      # FastAPI backend
│   │   ├── main.py               # App entry point + lifespan
│   │   ├── routes/               # Endpoint handlers — one module per logical group
│   │   │   ├── series.py         # GET /v1/series
│   │   │   ├── forecast.py       # GET /v1/forecast/{id}
│   │   │   ├── pricing.py        # GET /v1/recommend/{id}, POST /v1/batch-recommend
│   │   │   ├── explain.py        # GET /v1/explain/{id}, GET /v1/uncertainty/{id}
│   │   │   └── simulate.py       # POST /v1/simulate
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── auth.py               # X-API-Key authentication (optional)
│   │   └── rate_limit.py         # In-process sliding-window rate limiter
│   ├── dashboard/                # Streamlit frontend
│   │   ├── app.py                # Page layout + tabs
│   │   └── charts.py             # Reusable Plotly figure builders
│   ├── data/
│   │   ├── loader.py             # DataStore: loads all artefacts at startup
│   │   ├── preprocessing.py      # Raw CSV loading and schema normalisation
│   │   └── features.py           # Full feature engineering pipeline (26 features)
│   ├── models/
│   │   ├── forecaster.py         # Nixtla forecast wrappers + ensemble weighting
│   │   ├── pricer.py             # LightGBM demand model + revenue grid search
│   │   └── uncertainty.py        # Conformal PIs + quantile pricing strategies
│   └── evaluation/
│       ├── metrics.py            # Forecast metrics (MAE, RMSE, MAPE, SMAPE)
│       └── explainability.py     # SHAP attribution + plain-English narratives
│
├── data/
│   ├── raw/avocado.csv           # Source dataset (Hass Avocado Board, 2015–2018)
│   ├── processed/                # Engineered features (generated by nb02)
│   └── outputs/                  # Model artefacts (generated by nb03–06)
│
├── tests/                        # Pytest test suite (256 unit + 36 integration)
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: 7 jobs — tests, lint, types, OpenAPI, Docker build
│       └── cd.yml                # CD: build + push to GHCR, Trivy scan, GitHub Release
├── Makefile                      # Developer targets — test, lint, lock, api, dashboard…
├── scripts/
│   ├── bootstrap.py              # Windows-friendly Makefile alternative
│   ├── export_openapi.py         # Regenerate docs/openapi.json
│   └── run_pipeline.py           # Execute notebooks 01–06 non-interactively
├── Dockerfile                    # API container (uv-based build)
├── Dockerfile.dashboard          # Dashboard container (uv-based build)
├── docker-compose.yml            # Orchestration
├── requirements.txt              # Full dependency list (ML stack)
├── requirements-ci.txt           # CI base deps (no heavy ML); lock: requirements-ci.lock
└── requirements-ci-integration.txt  # Integration test deps; lock: requirements-ci-integration.lock
```

---

## 🔌 API Reference

The REST API exposes seven business endpoints under the `/v1/` prefix, plus two
infrastructure endpoints at the root. Full interactive docs at `/docs` when running.

**Infrastructure (no auth, no version prefix)**

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check — series count, manifest integrity, artifact timestamp |
| `GET /metrics` | Prometheus text exposition (for scraping) |

**Business endpoints** (`/v1/` prefix — pass `X-API-Key` header when auth is enabled)

### `GET /v1/series`
List all 86 market series with region and product type metadata.
Supports `limit` and `offset` query parameters for pagination.

### `GET /v1/forecast/{unique_id}`
12-week price forecast for one series. Returns all 7 model predictions plus the
Ensemble_weighted 90% conformal prediction interval (lower and upper bounds).

```bash
curl http://localhost:8000/v1/forecast/Albany_conventional
```

### `GET /v1/recommend/{unique_id}`
Revenue-maximising price recommendation with elasticity estimate and revenue uplift projection.

```bash
curl http://localhost:8000/v1/recommend/LosAngeles_organic
```

### `GET /v1/explain/{unique_id}`
Top-3 SHAP demand drivers — which features are pushing predicted demand up or down right
now, each with its signed SHAP value, magnitude, and the actual feature value at the
current market context.

```bash
curl http://localhost:8000/v1/explain/Chicago_conventional
```

### `GET /v1/uncertainty/{unique_id}`
Three risk-profile pricing strategies (conservative / balanced / aggressive) derived from
P10, P50, and P90 quantile demand models. Includes risk metrics: downside risk %, uplift
Sharpe ratio, and whether all three strategies agree on price direction.

```bash
curl http://localhost:8000/v1/uncertainty/Chicago_conventional
```

### `POST /v1/batch-recommend`
Retrieve revenue-maximising recommendations for multiple series in one call. Unknown IDs
are returned in a `not_found` list rather than raising a 404, so valid IDs are always
served even when the request contains unrecognised ones.

```bash
curl -X POST http://localhost:8000/v1/batch-recommend \
  -H "Content-Type: application/json" \
  -d '{"unique_ids": ["Albany_conventional", "LosAngeles_organic"]}'
```

### `POST /v1/simulate`
Live LightGBM inference: swap in any price for a series and get the model's predicted
demand and revenue impact instantly. Useful for manual price experimentation before
committing to a recommendation.

> **Rate limit:** 30 requests / 60 s per IP address (configurable via
> `PRICING_SIMULATE_RATE_LIMIT`). Exceeding the limit returns HTTP 429 with a
> `Retry-After` header.

```bash
curl -X POST http://localhost:8000/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{"unique_id": "Albany_conventional", "price": 1.50}'
```

---

## 🧪 Tests

```bash
# Fast unit tests — no model artefacts required, runs in CI (256 tests)
make test
# or without make: pytest tests/ -m "not integration" -v

# Toy-DataStore integration tests — builds all artefacts from scratch using a
# lightweight LightGBM on synthetic data; requires lightgbm + scikit-learn (36 tests)
make test-integration
# or: pytest tests/test_integration.py -v

# Full test suite — requires running all notebooks first
make test-all
# or: pytest tests/ -v
```

> **Windows without GNU make:** `python scripts/bootstrap.py <target>` provides the same
> targets (`setup`, `test`, `test-integration`, `lint`, `api`, `dashboard`, `clean`, …).

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Forecasting** | StatsForecast (MSTL, AutoARIMA, AutoETS, AutoTheta), NeuralForecast (NHITS, NBEATSx, DLinear) |
| **Demand model** | LightGBM (gradient boosted trees) |
| **Uncertainty** | Conformal prediction (split conformal), Quantile LightGBM |
| **Explainability** | SHAP (TreeExplainer — exact Shapley values) |
| **API** | FastAPI + Uvicorn + Pydantic v2 |
| **Dashboard** | Streamlit + Plotly |
| **Containers** | Docker + docker-compose |
| **CI** | GitHub Actions |
| **Data** | pandas, NumPy, scikit-learn |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ☕ and avocado toast.
</div>
