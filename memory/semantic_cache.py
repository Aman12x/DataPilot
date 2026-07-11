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
import re
from hashlib import blake2b
from typing import Any

import numpy as np

from memory.store import _USE_PG, _connect, _db_path, init_db


def _hard_threshold() -> float:
    return float(os.getenv("SEMANTIC_CACHE_HARD_THRESHOLD", "0.92"))


def _soft_threshold() -> float:
    return float(os.getenv("SEMANTIC_CACHE_SOFT_THRESHOLD", "0.80"))


_EMBED_DIM = 384

_CANONICAL_TERMS = {
    "average": "average",
    "avg": "average",
    "mean": "average",
    "breakdown": "breakdown",
    "breakdowns": "breakdown",
    "group": "breakdown",
    "grouped": "breakdown",
    "groups": "breakdown",
    "salaries": "salary",
    "salary": "salary",
    "pay": "salary",
    "earned": "salary",
    "earn": "salary",
    "earnings": "salary",
    "departments": "department",
    "department": "department",
    "engineering": "department",
    "marketing": "department",
    "customers": "customer",
    "customer": "customer",
    "users": "user",
    "user": "user",
    "dau": "user",
    "daily": "user",
    "active": "user",
    "revenues": "revenue",
    "revenue": "revenue",
    "sales": "revenue",
    "income": "revenue",
    "forecast": "weather",
    "weather": "weather",
    "rain": "weather",
    "temperatures": "weather",
    "temperature": "weather",
    "umbrella": "weather",
    "patient": "health",
    "patients": "health",
    "readmission": "health",
    "readmissions": "health",
    "diabetes": "health",
    "cohort": "health",
    "cohorts": "health",
    "android": "android",
    "analyse": "analyze",
    "analyze": "analyze",
}

_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "had", "has", "have", "i", "in", "is", "it", "me", "of", "on", "or",
    "per", "please", "show", "should", "the", "this", "to", "was", "were",
    "what", "with",
}


class _FallbackEmbedder:
    """Deterministic local embedder used when MiniLM is not cached/available."""

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        vec = np.zeros(_EMBED_DIM, dtype=np.float32)
        lower = text.lower()
        tokens = re.findall(r"[a-z0-9]+", lower)
        features: list[str] = []
        for token in tokens:
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            token = _CANONICAL_TERMS.get(token, token)
            if token and token not in _FALLBACK_STOPWORDS:
                features.append(f"tok:{token}")

        # Add adjacent token pairs so short paraphrases like
        # "average revenue per customer" and "average revenue by customer"
        # remain very close without letting unrelated prose match by length.
        features.extend(
            f"bigram:{a}:{b}"
            for a, b in zip(features, features[1:])
        )

        for feature in features:
            digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % _EMBED_DIM
            weight = 2.0 if feature.startswith("tok:") else 1.0
            vec[idx] += weight

        norm = np.linalg.norm(vec)
        if normalize_embeddings and norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)


def _get_model():
    """Load MiniLM model (cached on first call), or a deterministic local fallback."""
    if not hasattr(_get_model, "_model"):
        try:
            from sentence_transformers import SentenceTransformer
            _get_model._model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                local_files_only=True,
            )
        except Exception:
            _get_model._model = _FallbackEmbedder()
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
    user_id: str | None = None,
    path: str | None = None,
) -> dict[str, Any] | None:
    """
    Look up the semantic cache for a (task, node_name, dataset_fingerprint, user_id) tuple.

    dataset_fingerprint scopes the cache to the specific dataset being analysed
    (e.g. the upload_id path for CSV uploads, or empty string for the demo DB).
    user_id scopes the cache per analyst so one user's results are never served
    to another user, even on the shared demo DB.
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

    if not user_id:
        return None

    query_vec = embed(task)

    params: list[Any] = [node_name, dataset_fingerprint]
    user_clause = ""
    if user_id:
        user_clause = "AND user_id = ?"
        params.append(user_id)

    with _connect(path) as con:
        rows = con.execute(
            f"""
            SELECT task_embedding, cached_result
            FROM   runs
            WHERE  cache_node_name     = ?
              AND  dataset_fingerprint = ?
              {user_clause}
              AND  task_embedding      IS NOT NULL
              AND  cached_result       IS NOT NULL
            ORDER  BY timestamp DESC
            """,
            tuple(params),
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
