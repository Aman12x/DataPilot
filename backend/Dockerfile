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

# Create persistent volume mount point for SQLite files
RUN mkdir -p /app/memory

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker: _runs and _run_owners are in-process dicts.
# For multi-worker scale-out, migrate to a Redis-backed run registry first.
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
