"""
memory/semantic_cache.py — Embedding-based cache using MiniLM + SQLite.

Skips LLM calls entirely when the current task is near-identical to a past run.
Uses sentence-transformers/all-MiniLM-L6-v2 (local, no API cost).

Thresholds (from .env):
  SEMANTIC_CACHE_HARD_THRESHOLD=0.92  → full cache hit, skip API call
  SEMANTIC_CACHE_SOFT_THRESHOLD=0.80  → soft hit, surface for analyst approval
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from memory.store import _USE_PG, _connect, _db_path, init_db


def _hard_threshold() -> float:
    return float(os.getenv("SEMANTIC_CACHE_HARD_THRESHOLD", "0.92"))


def _soft_threshold() -> float:
    return float(os.getenv("SEMANTIC_CACHE_SOFT_THRESHOLD", "0.80"))


def _get_model():
    """Load MiniLM model (cached on first call via module-level singleton)."""
    if not hasattr(_get_model, "_model"):
        from sentence_transformers import SentenceTransformer
        _get_model._model = SentenceTransformer("all-MiniLM-L6-v2")
    return _get_model._model


def embed(text: str) -> np.ndarray:
    """Return a normalised 384-dim embedding for the given text."""
    vec = _get_model().encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors (dot product)."""
    return float(np.dot(a, b))


def _ensure_cache_columns(path: str) -> None:
    """Add cache_node_name / cached_result / dataset_fingerprint columns to runs if not present."""
    blob_type = "BYTEA" if _USE_PG else "BLOB"
    with _connect(path) as con:
        if _USE_PG:
            cols = {
                row["column_name"]
                for row in con.execute(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_name = 'runs'"
                ).fetchall()
            }
        else:
            cols = {row["name"] for row in con.execute("PRAGMA table_info(runs)").fetchall()}
        if "cache_node_name" not in cols:
            con.execute("ALTER TABLE runs ADD COLUMN cache_node_name TEXT")
        if "cached_result" not in cols:
            con.execute(f"ALTER TABLE runs ADD COLUMN cached_result {blob_type}")
        if "dataset_fingerprint" not in cols:
            con.execute("ALTER TABLE runs ADD COLUMN dataset_fingerprint TEXT DEFAULT ''")


def check_cache(
    task: str,
    node_name: str,
    dataset_fingerprint: str = "",
    path: str | None = None,
) -> dict[str, Any] | None:
    """
    Look up the semantic cache for a (task, node_name, dataset_fingerprint) tuple.

    dataset_fingerprint scopes the cache to the specific dataset being analysed
    (e.g. the upload_id path for CSV uploads, or empty string for the demo DB).
    This prevents cross-dataset cache poisoning where an identical task string
    run against different data returns a cached result from the wrong dataset.

    Returns:
        None                                    — cache miss (similarity < soft threshold)
        {"hit_type": "hard", "result": dict}    — similarity >= hard threshold
        {"hit_type": "soft", "result": dict,
         "similarity": float}                   — soft threshold <= similarity < hard
    """
    path = path or _db_path()
    init_db(path)
    _ensure_cache_columns(path)

    query_vec = embed(task)

    with _connect(path) as con:
        rows = con.execute(
            """
            SELECT task_embedding, cached_result
            FROM   runs
            WHERE  cache_node_name    = ?
              AND  dataset_fingerprint = ?
              AND  task_embedding     IS NOT NULL
              AND  cached_result      IS NOT NULL
            ORDER  BY timestamp DESC
            """,
            (node_name, dataset_fingerprint),
        ).fetchall()

    best_sim    = 0.0
    best_result = None

    for row in rows:
        stored_vec = np.frombuffer(row["task_embedding"], dtype=np.float32)
        sim = cosine_similarity(query_vec, stored_vec)
        if sim > best_sim:
            best_sim    = sim
            best_result = json.loads(row["cached_result"])

    hard = _hard_threshold()
    soft = _soft_threshold()

    if best_result is None or best_sim < soft:
        return None
    if best_sim >= hard:
        return {"hit_type": "hard", "result": best_result, "similarity": best_sim}
    return {"hit_type": "soft", "result": best_result, "similarity": best_sim}


def store_cache(
    task: str,
    node_name: str,
    result: dict[str, Any],
    run_id: str,
    dataset_fingerprint: str = "",
    path: str | None = None,
) -> None:
    """
    Persist the embedding + result for a completed LLM call so future runs
    can hit the cache.

    Args:
        task:                The task string that was analysed.
        node_name:           The graph node that produced the result (e.g. 'generate_sql').
        result:              The dict result to cache.
        run_id:              The run_id of the row to update (must already exist in runs).
        dataset_fingerprint: Scopes the cache to a specific dataset (upload_id path or '').
        path:                Optional DB path override.
    """
    path = path or _db_path()
    init_db(path)
    _ensure_cache_columns(path)

    vec_bytes    = embed(task).tobytes()
    result_bytes = json.dumps(result)

    with _connect(path) as con:
        con.execute(
            """
            UPDATE runs
               SET task_embedding      = ?,
                   cache_node_name     = ?,
                   cached_result       = ?,
                   dataset_fingerprint = ?
             WHERE run_id = ?
            """,
            (vec_bytes, node_name, result_bytes, dataset_fingerprint, run_id),
        )
