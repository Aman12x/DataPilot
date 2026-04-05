"""
memory/store.py — Run logger for DataPilot.

Logs every completed graph run: task, params, analyst overrides, eval score,
token usage, and cost. Acts as the self-improvement substrate — past runs are
queried by retriever.py and semantic_cache.py.

Runs are scoped to user_id so each analyst only sees their own history.

Storage backend (in priority order):
  1. DATABASE_URL set → PostgreSQL (psycopg2)
  2. MEMORY_DB_PATH / default → SQLite
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator


def _db_path() -> str:
    return os.getenv("MEMORY_DB_PATH", "memory/datapilot_memory.db")


# ── Postgres / SQLite abstraction ─────────────────────────────────────────────

_DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_PG = bool(_DATABASE_URL)


def _q(sql: str) -> str:
    """Translate SQLite ? placeholders to Postgres %s."""
    return sql.replace("?", "%s") if _USE_PG else sql


class _PGCursorAdapter:
    """Wraps a psycopg2 RealDictCursor to match the sqlite3 cursor interface."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()


class _PGConnAdapter:
    """Wraps a psycopg2 connection to expose sqlite3-style .execute()."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()) -> _PGCursorAdapter:
        import psycopg2.extras  # noqa: PLC0415

        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_q(sql), params)
        return _PGCursorAdapter(cur)


@contextmanager
def _connect(path: str) -> Generator[Any, None, None]:
    """
    Yield a database connection for *path* (SQLite) or DATABASE_URL (Postgres).

    The caller uses ``with _connect(path) as con:`` — identical to the old
    sqlite3 context-manager pattern but now routes to Postgres when
    DATABASE_URL is set.
    """
    if _USE_PG:
        import psycopg2  # noqa: PLC0415

        conn = psycopg2.connect(_DATABASE_URL)
        try:
            yield _PGConnAdapter(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db(path: str | None = None) -> None:
    """Create the runs table if it doesn't exist, and migrate if needed."""
    path = path or _db_path()
    # BLOB in SQLite, BYTEA in Postgres
    blob_type = "BYTEA" if _USE_PG else "BLOB"
    with _connect(path) as con:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS runs (
                run_id               TEXT PRIMARY KEY,
                timestamp            TEXT,
                task                 TEXT,
                task_embedding       {blob_type},
                metric               TEXT,
                covariate            TEXT,
                db_backend           TEXT,
                analyst_override     TEXT,
                top_segment          TEXT,
                eval_score           REAL,
                cache_read_tokens    INTEGER,
                cache_write_tokens   INTEGER,
                uncached_tokens      INTEGER,
                semantic_cache_hits  INTEGER,
                estimated_cost_usd   REAL,
                notes                TEXT,
                user_id              TEXT,
                analysis_mode        TEXT
            )
        """)

        # Incremental migrations — check existing columns before ALTER TABLE
        # to avoid transaction-aborting errors on already-migrated databases.
        if _USE_PG:
            existing = {
                row["column_name"]
                for row in con.execute(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_name = 'runs'"
                ).fetchall()
            }
        else:
            existing = {
                row["name"]
                for row in con.execute("PRAGMA table_info(runs)").fetchall()
            }

        for col_name, col_defn in [
            ("user_id",       "TEXT"),
            ("analysis_mode", "TEXT"),
            ("audit_passed",  "INTEGER DEFAULT 0"),
        ]:
            if col_name not in existing:
                con.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_defn}")


def log_run(
    task: str,
    *,
    path: str | None = None,
    run_id: str | None = None,
    user_id: str | None = None,
    analysis_mode: str = "ab_test",
    metric: str = "",
    covariate: str = "",
    db_backend: str = "duckdb",
    analyst_override: dict[str, Any] | None = None,
    top_segment: str = "",
    eval_score: float | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    uncached_tokens: int = 0,
    semantic_cache_hits: int = 0,
    estimated_cost_usd: float = 0.0,
    task_embedding: bytes | None = None,
    notes: str = "",
    audit_passed: bool = False,
) -> str:
    """
    Persist one run to the memory store.
    Returns the run_id (auto-generated UUID if not provided).
    """
    path = path or _db_path()
    run_id = run_id or str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    init_db(path)

    override_json = json.dumps(analyst_override) if analyst_override else None

    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id, timestamp, task, task_embedding, metric, covariate,
                db_backend, analyst_override, top_segment, eval_score,
                cache_read_tokens, cache_write_tokens, uncached_tokens,
                semantic_cache_hits, estimated_cost_usd, notes, user_id, analysis_mode,
                audit_passed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, ts, task, task_embedding, metric, covariate,
                db_backend, override_json, top_segment, eval_score,
                cache_read_tokens, cache_write_tokens, uncached_tokens,
                semantic_cache_hits, estimated_cost_usd, notes, user_id, analysis_mode,
                int(audit_passed),
            ),
        )
    return run_id


def update_eval_score(run_id: str, eval_score: float, path: str | None = None) -> None:
    path = path or _db_path()
    init_db(path)
    with _connect(path) as con:
        con.execute(
            "UPDATE runs SET eval_score = ? WHERE run_id = ?",
            (eval_score, run_id),
        )


def get_run(run_id: str, path: str | None = None) -> dict[str, Any] | None:
    path = path or _db_path()
    init_db(path)
    with _connect(path) as con:
        row = con.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("analyst_override"):
        d["analyst_override"] = json.loads(d["analyst_override"])
    return d


def get_all_runs(
    path: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Return runs ordered by timestamp descending.

    Args:
        user_id: When provided, return only this user's runs.
                 When None, return all runs (used by eval harness).
        limit:   Maximum number of runs to return.
    """
    path = path or _db_path()
    init_db(path)
    with _connect(path) as con:
        if user_id:
            rows = con.execute(
                "SELECT * FROM runs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("analyst_override"):
            d["analyst_override"] = json.loads(d["analyst_override"])
        result.append(d)
    return result
