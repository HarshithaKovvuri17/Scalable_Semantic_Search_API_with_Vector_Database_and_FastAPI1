# ==============================================================================
# Production Dockerfile for Scalable Semantic Search API (Optimized for Speed)
# ==============================================================================
FROM python:3.10-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - libgomp1: required by FAISS (OpenMP runtime)
# - curl: required for Docker container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-optimized PyTorch first (~150MB vs ~900MB GPU PyTorch) to drastically accelerate Docker build
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy entrypoint script and set executable permissions
COPY docker-entrypoint.sh .
RUN chmod +x /app/docker-entrypoint.sh

# Copy application source code, dataset, models, scripts, and tests
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY .env.example .env.example
COPY .env .env

# Cache model for container: reuse local cached model if present; otherwise download
RUN python -c "from pathlib import Path; import shutil; p = Path('/app/models/sentence_transformer'); \
((from_hf := __import__('sentence_transformers').SentenceTransformer)('sentence-transformers/all-MiniLM-L6-v2').save('/app/models/sentence_transformer')) if not (p/'config.json').exists() else None; \
shutil.copytree('/app/models/sentence_transformer', '/app/models_cache/sentence_transformer', dirs_exist_ok=True)"

# Expose API port
EXPOSE 8000

# Fast container healthcheck against /health endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint to verify data readiness
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default server startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
