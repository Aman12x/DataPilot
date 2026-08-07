FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Download MiniLM embedding model into the image layer so cold starts
# don't hit HuggingFace at runtime (eliminates ~10s of HTTP requests).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire project (agents/, tools/, memory/, auth/, etc.)
COPY . /app

# The persistent volume mounts at /app/db, NOT here — see railway.toml.
# `memory/` is a Python package (store.py, semantic_cache.py, retriever.py) that
# the COPY above already provides; mounting a volume over it shadows the package
# and the backend dies at startup with
#   ImportError: cannot import name 'retriever' from 'memory' (unknown location)
# Railway creates /app/db itself when the volume is attached, so there is
# nothing to mkdir here.

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker: _runs and _run_owners are in-process dicts.
# For multi-worker scale-out, migrate to a Redis-backed run registry first.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
