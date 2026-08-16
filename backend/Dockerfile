FROM python:3.13-slim

WORKDIR /app

# Install system deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Unprivileged runtime user. Fixed uid so the entrypoint's volume chown and
# any platform-side uid setting (RAILWAY_RUN_UID) agree on a number.
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

# Download MiniLM embedding model into the image layer so cold starts
# don't hit HuggingFace at runtime (eliminates ~10s of HTTP requests).
# HF_HOME must be set *before* this and stay set at runtime: the model is
# loaded with local_files_only=True and silently falls back to a hashing
# embedder if the cache is not where huggingface_hub looks (the default is
# ~/.cache, which differs between the build-time root and the runtime user).
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire project (agents/, tools/, memory/, auth/, etc.)
COPY --chown=app:app . /app
# The model cache above was written by root; the first-boot sample dataset
# and default SQLite paths also live under /app.
RUN chown -R app:app /app/.cache && chmod +x /app/backend/entrypoint.sh

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

# Starts as root only long enough to chown the volume mount, then drops to
# `app` (see backend/entrypoint.sh). No USER directive on purpose: the
# entrypoint is what makes the volume writable by the unprivileged user.
ENTRYPOINT ["/app/backend/entrypoint.sh"]

# Single worker: _runs and _run_owners are in-process dicts.
# For multi-worker scale-out, migrate to a Redis-backed run registry first.
CMD ["sh", "-c", "exec uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
