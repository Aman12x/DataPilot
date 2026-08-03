"""
memory/retriever.py — Query past runs for history injection.

Takes the current task string, finds the top-N most relevant past runs by
keyword overlap, and returns them for injection into the agent's system prompt.
Keyword overlap is cheap and fast — semantic similarity lives in semantic_cache.py.

Phase 2: prefer same metric_pack_id / connection_id when retrieving.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from memory.store import _connect, _db_path, init_db

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, length >= 3."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def _overlap_score(task_tokens: set[str], run: dict[str, Any]) -> int:
    """Count shared tokens between current task and a past run's task + top_segment."""
    run_text  = f"{run.get('task', '')} {run.get('top_segment', '')} {run.get('metric', '')}"
    run_tokens = _tokenize(run_text)
    return len(task_tokens & run_tokens)


def _scope_boost(run: dict[str, Any], pack_id: str | None, connection_id: str | None) -> float:
    """Prefer same pack, then same connection, over cross-scope history."""
    boost = 0.0
    if pack_id and run.get("metric_pack_id") == pack_id:
        boost += 2.0
    if connection_id and run.get("connection_id") == connection_id:
        boost += 1.0
    # Mild penalty when a pack-scoped run is pulled into a different pack
    if pack_id and run.get("metric_pack_id") and run.get("metric_pack_id") != pack_id:
        boost -= 1.5
    if connection_id and run.get("connection_id") and run.get("connection_id") != connection_id:
        boost -= 1.0
    return boost


def retrieve_relevant_history(
    task: str,
    top_n: int = 3,
    path: str | None = None,
    user_id: str | None = None,
    metric_pack_id: str | None = None,
    connection_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return the top-N most relevant past runs for the given task, ranked by
    keyword overlap (+ pack/connection boost). Scoped to user_id when provided.
    """
    path = path or _db_path()
    init_db(path)
    with _connect(path) as con:
        if user_id:
            rows = con.execute(
                """SELECT run_id, task, metric, top_segment,
                          analyst_override, eval_score, timestamp,
                          metric_pack_id, connection_id
                   FROM   runs
                   WHERE  user_id = ? AND audit_passed = 1
                   ORDER  BY timestamp DESC LIMIT 80""",
                (user_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT run_id, task, metric, top_segment,
                          analyst_override, eval_score, timestamp,
                          metric_pack_id, connection_id
                   FROM   runs
                   WHERE  audit_passed = 1
                   ORDER  BY timestamp DESC LIMIT 80""",
            ).fetchall()
    all_runs = [dict(r) for r in rows]

    if not all_runs:
        return []

    task_tokens = _tokenize(task)

    def _combined_score(run: dict[str, Any]) -> float:
        overlap = _overlap_score(task_tokens, run)
        quality = run.get("eval_score") or 0.5
        return overlap * 0.7 + quality * 0.3 + _scope_boost(run, metric_pack_id, connection_id)

    scored = [(run, _combined_score(run)) for run in all_runs]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = [run for run, score in scored[:top_n] if _overlap_score(task_tokens, run) > 0]

    return [
        {
            "run_id":            r["run_id"],
            "task":              r["task"],
            "metric":            r["metric"],
            "top_segment":       r["top_segment"],
            "analyst_override":  r["analyst_override"],
            "eval_score":        r["eval_score"],
            "timestamp":         r["timestamp"],
            "metric_pack_id":    r.get("metric_pack_id"),
            "connection_id":     r.get("connection_id"),
        }
        for r in top
    ]


def retrieve_sql_examples(
    task: str,
    top_n: int = 2,
    min_similarity: float = 0.40,
    path: str | None = None,
    user_id: str | None = None,
    metric_pack_id: str | None = None,
    connection_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve verified question-SQL pairs ranked by embedding similarity.

    When metric_pack_id / connection_id are set, same-scope examples are
    preferred (score boost). Cross-connection examples are still allowed
    but demoted — `_filter_few_shot_by_schema` removes incompatible tables.
    """
    path = path or _db_path()
    init_db(path)
    # The cache columns this query reads are added by a migration that
    # otherwise only runs on cache check/store. On a fresh database, relying
    # on node order to have migrated first is fragile — run it here too.
    try:
        from memory.semantic_cache import _ensure_cache_columns
        _ensure_cache_columns(path)
    except Exception:
        return []

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
                SELECT task, task_embedding, cached_result, metric_pack_id, connection_id
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
                SELECT task, task_embedding, cached_result, metric_pack_id, connection_id
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
            d = dict(row)
            stored_vec = np.frombuffer(d["task_embedding"], dtype=np.float32)
            sim = cosine_similarity(query_vec, stored_vec)
            sim += 0.05 * _scope_boost(d, metric_pack_id, connection_id)
            if sim < min_similarity:
                continue
            result = json.loads(d["cached_result"])
            sql = result.get("sql", "").strip()
            if sql:
                scored.append({
                    "task": d["task"],
                    "sql": sql,
                    "similarity": float(sim),
                    "metric_pack_id": d.get("metric_pack_id"),
                    "connection_id": d.get("connection_id"),
                })
        except Exception as exc:
            logger.debug("retrieve_sql_examples: skipping malformed cache row: %s", exc)
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]
