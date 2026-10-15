FROM python:3.12-slim

WORKDIR /app

# System deps — only what's needed for building C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps — split into two layers so heavy packages cache
COPY backend/requirements.txt /tmp/requirements.txt

# Install lighter deps first (fast layer)
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    "pydantic>=2.9" \
    pydantic-settings==2.5.2 \
    sqlalchemy==2.0.35 \
    python-multipart==0.0.12 \
    aiofiles==24.1.0 \
    joblib==1.4.2

# Install heavy ML deps (slower layer, cached separately)
# Note: xgboost Linux wheel includes CUDA/nccl by default; use --no-deps
# and install only required runtime deps (scipy, numpy) separately
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    "numpy>=1.26.4" \
    pyarrow==17.0.0 \
    scikit-learn==1.5.2 \
    optuna==3.6.1
# Install xgboost CPU-only (skip the nvidia-nccl-cu12 optional GPU dep)
RUN pip install --no-cache-dir --no-deps xgboost==2.1.1 && \
    pip install --no-cache-dir scipy

# Install SHAP — 0.52+ (Python 3.12) removed numba/CUDA requirement, uses nanobind
RUN pip install --no-cache-dir "shap>=0.52.0"

# Copy application source
COPY backend/ /app/backend/
COPY ml/src/  /app/ml/src/

# Model artifacts mounted at runtime via docker-compose volume
WORKDIR /app/backend

ENV PYTHONPATH=/app/backend:/app/ml/src
ENV DATABASE_URL=sqlite:////app/data/aml_priority.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
