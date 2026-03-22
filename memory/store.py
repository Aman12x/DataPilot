"""
memory/store.py — SQLite run logger.

Logs every completed graph run: task, params, analyst overrides, eval score,
token usage, and cost. Acts as the self-improvement substrate — past runs are
queried by retriever.py and semantic_cache.py.

Runs are scoped to user_id so each analyst only sees their own history.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any


def _db_path() -> str:
    return os.getenv("MEMORY_DB_PATH", "memory/datapilot_memory.db")


def _connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_db(path: str | None = None) -> None:
    """Create the runs table if it doesn't exist, and migrate if needed."""
    path = path or _db_path()
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id               TEXT PRIMARY KEY,
                timestamp            TEXT,
                task                 TEXT,
                task_embedding       BLOB,
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
        # Incremental migrations for existing DBs
        existing = {row[1] for row in con.execute("PRAGMA table_info(runs)").fetchall()}
        for col, defn in [
            ("user_id",       "TEXT"),
            ("analysis_mode", "TEXT"),
        ]:
            if col not in existing:
                con.execute(f"ALTER TABLE runs ADD COLUMN {col} {defn}")


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
) -> str:
    """
    Persist one run to the memory store.
    Returns the run_id (auto-generated UUID if not provided).
    """
    path   = path or _db_path()
    run_id = run_id or str(uuid.uuid4())
    ts     = datetime.now(timezone.utc).isoformat()

    init_db(path)

    override_json = json.dumps(analyst_override) if analyst_override else None

    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id, timestamp, task, task_embedding, metric, covariate,
                db_backend, analyst_override, top_segment, eval_score,
                cache_read_tokens, cache_write_tokens, uncached_tokens,
                semantic_cache_hits, estimated_cost_usd, notes, user_id, analysis_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, ts, task, task_embedding, metric, covariate,
                db_backend, override_json, top_segment, eval_score,
                cache_read_tokens, cache_write_tokens, uncached_tokens,
                semantic_cache_hits, estimated_cost_usd, notes, user_id, analysis_mode,
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
