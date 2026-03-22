"""
memory/retriever.py — Query past runs for history injection.

Takes the current task string, finds the top-N most relevant past runs by
keyword overlap, and returns them for injection into the agent's system prompt.
Keyword overlap is cheap and fast — semantic similarity lives in semantic_cache.py.
"""

from __future__ import annotations

import pickle
import re
from typing import Any

import numpy as np

from memory.store import _connect, _db_path, get_all_runs, init_db


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, length >= 3."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def _overlap_score(task_tokens: set[str], run: dict[str, Any]) -> int:
    """Count shared tokens between current task and a past run's task + top_segment."""
    run_text  = f"{run.get('task', '')} {run.get('top_segment', '')} {run.get('metric', '')}"
    run_tokens = _tokenize(run_text)
    return len(task_tokens & run_tokens)


def retrieve_relevant_history(
    task: str,
    top_n: int = 3,
    path: str | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return the top-N most relevant past runs for the given task, ranked by
    keyword overlap. Scoped to user_id when provided.

    Args:
        task:    Current analyst task string.
        top_n:   Maximum number of runs to return.
        path:    Optional override for the SQLite DB path.
        user_id: When set, only considers runs belonging to this user.

    Returns:
        List of run dicts (subset of fields useful for prompt injection):
        [{run_id, task, metric, top_segment, analyst_override, eval_score, timestamp}]
    """
    init_db(path)
    all_runs = get_all_runs(path, user_id=user_id)

    if not all_runs:
        return []

    task_tokens = _tokenize(task)

    scored = [
        (run, _overlap_score(task_tokens, run))
        for run in all_runs
    ]
    # Sort: descending overlap score, then descending timestamp (already ordered)
    scored.sort(key=lambda x: x[1], reverse=True)

    top = [run for run, score in scored[:top_n] if score > 0]

    # Return only the fields useful for history injection
    return [
        {
            "run_id":            r["run_id"],
            "task":              r["task"],
            "metric":            r["metric"],
            "top_segment":       r["top_segment"],
            "analyst_override":  r["analyst_override"],
            "eval_score":        r["eval_score"],
            "timestamp":         r["timestamp"],
        }
        for r in top
    ]


def retrieve_sql_examples(
    task: str,
    top_n: int = 2,
    min_similarity: float = 0.40,
    path: str | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve verified question-SQL pairs from the memory store, ranked by
    embedding similarity to the current task.

    Used for few-shot injection into SQL_GENERATION_PROMPT: including similar
    past (question, SQL) pairs as in-context examples consistently improves
    accuracy on BIRD and Spider benchmarks (DAIL-SQL pattern).

    Args:
        task:            Current analyst task string.
        top_n:           Maximum number of examples to return.
        min_similarity:  Minimum cosine similarity threshold (0.40 = loosely related).
                         Lower than semantic-cache thresholds (0.80/0.92) because
                         even loosely related examples help the model understand
                         the schema and query structure.
        path:            Optional DB path override.

    Returns:
        List of dicts: [{"task": str, "sql": str, "similarity": float}]
        Sorted by similarity descending.  Empty list if no examples exist or
        sentence-transformers is not installed.
    """
    path = path or _db_path()
    init_db(path)

    # Lazy import — semantic_cache carries the sentence-transformers dependency.
    # Return empty list gracefully if the model isn't available.
    try:
        from memory.semantic_cache import cosine_similarity, embed
    except Exception:
        return []

    try:
        query_vec = embed(task)
    except Exception:
        return []

    with _connect(path) as con:
        if user_id:
            rows = con.execute(
                """
                SELECT task, task_embedding, cached_result
                FROM   runs
                WHERE  cache_node_name = 'generate_sql'
                  AND  task_embedding  IS NOT NULL
                  AND  cached_result   IS NOT NULL
                  AND  user_id         = ?
                ORDER  BY timestamp DESC
                LIMIT  200
                """,
                (user_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT task, task_embedding, cached_result
                FROM   runs
                WHERE  cache_node_name = 'generate_sql'
                  AND  task_embedding  IS NOT NULL
                  AND  cached_result   IS NOT NULL
                ORDER  BY timestamp DESC
                LIMIT  200
                """,
            ).fetchall()

    scored: list[dict[str, Any]] = []
    for row in rows:
        try:
            stored_vec = np.frombuffer(row["task_embedding"], dtype=np.float32)
            sim = cosine_similarity(query_vec, stored_vec)
            if sim < min_similarity:
                continue
            result = pickle.loads(row["cached_result"])  # noqa: S301
            sql = result.get("sql", "").strip()
            if sql:
                scored.append({"task": row["task"], "sql": sql, "similarity": sim})
        except Exception:
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]
