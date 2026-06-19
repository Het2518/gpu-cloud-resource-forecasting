FROM python:3.11-slim

WORKDIR /app

# Install only non-torch dependencies for the lightweight Docker image
# PyTorch is pre-installed in the host environment and mounted via volume
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only torch (much smaller — avoids CUDA download in Docker)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies (no torch/cuda re-download)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    python-multipart>=0.0.9 \
    pydantic>=2.6.0 \
    scikit-learn>=1.3.0 \
    joblib>=1.3.0 \
    numpy>=1.24.0 \
    polars>=0.20.0 \
    pyarrow>=14.0.0 \
    prometheus-fastapi-instrumentator>=7.0.0 \
    python-dotenv>=1.0.0 \
    mlflow>=2.13.0 \
    gunicorn>=21.2.0

# Copy application source
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["gunicorn", "main:app", "--workers", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
