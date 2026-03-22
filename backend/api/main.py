"""
backend/api/main.py — FastAPI application entry point.

Checkpointer selection (in priority order):
  1. DATABASE_URL set → PostgreSQL (langgraph-checkpoint-postgres)
  2. GRAPH_DB_PATH set / default → SQLite (langgraph-checkpoint-sqlite)

Redis (optional, REDIS_URL):
  When set, run state and rate limits are Redis-backed (multi-pod safe).
  Without it, in-memory queues are used (single-pod, fine for dev).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _prewarm_embedder() -> None:
    """Load the MiniLM model into memory once at startup (no-op if unavailable)."""
    try:
        from memory.semantic_cache import embed
        embed("warmup")
    except Exception:
        pass

# Sentry — opt-in via SENTRY_DSN env var. No-op when unset.
_SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            traces_sample_rate=0.1,
            environment=os.getenv("RAILWAY_ENVIRONMENT", "production"),
            release=os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown"),
        )
        logger.info("Sentry initialised (environment=%s)", os.getenv("RAILWAY_ENVIRONMENT", "production"))
    except ImportError:
        logger.warning("SENTRY_DSN set but sentry-sdk not installed — run: pip install sentry-sdk[fastapi]")


def _make_sqlite_checkpointer():
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from agents.analyze.graph import _PickleSerde

    db_path = os.getenv("GRAPH_DB_PATH", "memory/graph.db")
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # WAL mode: concurrent reads don't block writes; no deadlocks under load.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.commit()
    logger.info("Using SQLite checkpointer at %s (WAL mode)", db_path)
    return SqliteSaver(conn, serde=_PickleSerde())


async def _make_postgres_checkpointer(database_url: str):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
    await checkpointer.setup()
    logger.info("Using PostgreSQL checkpointer")
    return checkpointer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Redis ─────────────────────────────────────────────────────────────────
    from .run_manager import REDIS_URL, set_redis_client
    redis_client = None
    if REDIS_URL:
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)
            await redis_client.ping()
            set_redis_client(redis_client)
            logger.info("Redis connected (%s)", REDIS_URL.split("@")[-1])
        except Exception as exc:
            logger.warning("Redis connection failed (%s) — using in-memory queues", exc)
            redis_client = None
    else:
        logger.info("REDIS_URL not set — using in-memory run queues (single-pod only)")

    # ── Checkpointer ─────────────────────────────────────────────────────────
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            checkpointer = await _make_postgres_checkpointer(database_url)
        except Exception as exc:
            logger.warning("PostgreSQL checkpointer failed (%s), falling back to SQLite", exc)
            checkpointer = _make_sqlite_checkpointer()
    else:
        checkpointer = _make_sqlite_checkpointer()

    # ── Auth DB WAL mode ──────────────────────────────────────────────────────
    try:
        import sqlite3 as _sqlite3
        auth_path = os.getenv("AUTH_DB_PATH", "memory/auth.db")
        os.makedirs(os.path.dirname(auth_path) or ".", exist_ok=True)
        _ac = _sqlite3.connect(auth_path)
        _ac.execute("PRAGMA journal_mode=WAL")
        _ac.execute("PRAGMA busy_timeout=5000")
        _ac.commit()
        _ac.close()
    except Exception as exc:
        logger.warning("Could not set WAL on auth.db: %s", exc)

    # ── Graph ─────────────────────────────────────────────────────────────────
    from agents.analyze.graph import build_graph
    app.state.graph = build_graph(checkpointer)

    # ── Memory store ─────────────────────────────────────────────────────────
    import memory.store as _mem_store
    app.state.memory_store = _mem_store

    # ── Upload directory ──────────────────────────────────────────────────────
    upload_dir = os.getenv("UPLOAD_DIR", "tmp_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # ── DuckDB + embedding warm-up run in background ──────────────────────────
    # These are slow (~30s total) but not needed for the /health check.
    # Running them as a background task lets uvicorn accept requests immediately
    # so Railway's healthcheck passes while data generation continues.
    import asyncio

    async def _background_init():
        try:
            import runpy
            data_script = os.path.join(_PROJECT_ROOT, "data", "generate_data.py")
            db_file     = os.path.join(_PROJECT_ROOT, "data", "dau_experiment.db")
            if os.path.exists(data_script) and not os.path.exists(db_file):
                await asyncio.to_thread(
                    runpy.run_path, data_script, run_name="__main__"
                )
                logger.info("DuckDB sample data generated")
            else:
                logger.info("DuckDB sample data ready")
        except Exception as exc:
            logger.warning("Could not generate DuckDB data: %s", exc)
        try:
            await asyncio.to_thread(_prewarm_embedder)
            logger.info("Embedding model pre-warmed")
        except Exception as exc:
            logger.warning("Could not pre-warm embedding model: %s", exc)

    asyncio.create_task(_background_init())

    # ── Upload TTL sweeper ─────────────────────────────────────────────────────
    _UPLOAD_TTL_HOURS   = float(os.getenv("UPLOAD_TTL_HOURS", "24"))
    _SWEEP_INTERVAL_SEC = float(os.getenv("UPLOAD_SWEEP_INTERVAL_SEC", "3600"))

    async def _sweep_uploads():
        cutoff_sec = _UPLOAD_TTL_HOURS * 3600
        root = Path(upload_dir)
        while True:
            await asyncio.sleep(_SWEEP_INTERVAL_SEC)
            try:
                now     = time.time()
                deleted = 0
                for db_file in root.glob("*/*.db"):
                    try:
                        age = now - db_file.stat().st_mtime
                        if age > cutoff_sec:
                            db_file.unlink(missing_ok=True)
                            deleted += 1
                    except Exception as exc:
                        logger.debug("sweep: could not remove %s — %s", db_file, exc)
                if deleted:
                    logger.info("Upload sweeper removed %d expired file(s) from %s", deleted, upload_dir)
            except Exception as exc:
                logger.warning("Upload sweeper error: %s", exc)

    _sweep_task = asyncio.create_task(_sweep_uploads())

    yield

    # ── Cleanup ───────────────────────────────────────────────────────────────
    _sweep_task.cancel()
    if redis_client:
        await redis_client.aclose()
    logger.info("DataPilot backend shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="DataPilot API", version="1.0.0", lifespan=lifespan)

# CORS: in prod set CORS_ORIGINS=https://your-frontend.railway.app (comma-separated).
# Without it the API is open to all origins — fine for dev, not for prod.
_cors_raw = os.getenv("CORS_ORIGINS", "")
_origins  = [o.strip() for o in _cors_raw.split(",") if o.strip()]
_wildcard = not _origins
if _wildcard:
    logger.warning("CORS_ORIGINS not set — all origins allowed. Set in production.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _wildcard else _origins,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes.auth import router as auth_router
from .routes.runs import router as runs_router
from .routes.upload import router as upload_router
from .routes.samples import router as samples_router

app.include_router(auth_router)
app.include_router(runs_router)
app.include_router(upload_router)
app.include_router(samples_router)


@app.get("/health")
def health():
    return {"status": "ok"}
