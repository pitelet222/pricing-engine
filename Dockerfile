# ---- build stage: install heavy deps once and cache them ----
FROM python:3.11-slim AS builder

WORKDIR /build

# uv: fast Python package installer (replaces pip; ~10-100x faster for large stacks)
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv

# System deps required by LightGBM, scikit-learn, and PyTorch build chains
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt


# ---- runtime stage ----
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code (data/ is mounted as a volume at runtime, not baked in)
COPY src/ ./src/

# Hand ownership to the non-root user before switching
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Data artifacts (data/outputs/, data/processed/) must be mounted at runtime.
# Example: docker run -v $(pwd)/data:/app/data:ro ...
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
