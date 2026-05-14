# ---- build stage: install heavy deps once and cache them ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps required by LightGBM, scikit-learn, and PyTorch build chains
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ---- runtime stage ----
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code (data/ is mounted as a volume at runtime, not baked in)
COPY src/ ./src/

EXPOSE 8000

# Data artifacts (data/outputs/, data/processed/) must be mounted at runtime.
# Example: docker run -v $(pwd)/data:/app/data ...
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
