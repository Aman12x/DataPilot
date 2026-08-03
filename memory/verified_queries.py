"""
memory/verified_queries.py — the verified-query repository (future-work item 6).

Gate-approved SQL used to evaporate after each run; this table keeps it. Two
intake paths, one repository:

  gate         SQL a human approved at the query gate, stored automatically
               when the run completes. Every gated run is a free
               human-verified question→SQL pair.
  contributed  Canonical queries a user added deliberately — the "teach it
               how we write queries" path.

Rows are keyed by task embedding, scoped to (workspace | user, connection),
and stamped with the schema hash they were verified against. Retrieval feeds
`generate_sql`'s few-shot block: contributed exemplars outrank gate captures,
which outrank incidental cache hits; a schema-hash mismatch demotes an
example (the final stale-table guard is `_filter_few_shot_by_schema`).

Deliberately small: contributed queries are capped per scope. Raw retrieval
over thousands of historical queries is known to move accuracy by less than
a point — curated exemplars are the lever, so the cap forces curation.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from memory.store import _connect, _db_path, _USE_PG

logger = logging.getLogger(__name__)

CONTRIBUTED_CAP = int(os.getenv("VERIFIED_QUERY_CAP", "20"))

# Score shaping for retrieval. Contributed > gate > (elsewhere) cache rows,
# and an example verified against a different schema is pushed down hard.
_SOURCE_BOOST = {"contributed": 0.10, "gate": 0.05}
_SCHEMA_MISMATCH_PENALTY = 0.15


def init_verified_queries(path: str | None = None) -> None:
    path = path or _db_path()
    blob_type = "BYTEA" if _USE_PG else "BLOB"
    with _connect(path) as con:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS verified_queries (
                vq_id           TEXT PRIMARY KEY,
                source          TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                workspace_id    TEXT DEFAULT '',
                name            TEXT DEFAULT '',
                task            TEXT NOT NULL,
                sql             TEXT NOT NULL,
                task_embedding  {blob_type},
                connection_id   TEXT DEFAULT '',
                schema_hash     TEXT DEFAULT '',
                created_at      TEXT
            )
        """)


def _embed(task: str) -> bytes | None:
    try:
        from memory.semantic_cache import embed
        return embed(task).tobytes()
    except Exception as exc:  # noqa: BLE001 — embedding is best-effort
        logger.debug("verified_queries: embed failed: %s", exc)
        return None


def _scope_clause(user_id: str, workspace_id: str) -> tuple[str, tuple]:
    """Rows visible to a caller: their own, plus their workspace's."""
    if workspace_id:
        return "(user_id = ? OR workspace_id = ?)", (user_id, workspace_id)
    return "user_id = ?", (user_id,)


def add_verified_query(
    task: str,
    sql: str,
    *,
    source: str,
    user_id: str,
    workspace_id: str = "",
    name: str = "",
    connection_id: str = "",
    schema_hash: str = "",
    path: str | None = None,
) -> str:
    """Store a verified pair. Returns the new row's id.

    Gate intake upserts: a re-run of the same question against the same
    connection replaces the earlier capture instead of accumulating near
    duplicates. Contributed intake enforces the per-scope cap — exemplars,
    not a query log.
    """
    if source not in ("gate", "contributed"):
        raise ValueError(f"unknown source: {source}")
    if not task.strip() or not sql.strip():
        raise ValueError("task and sql are required")

    path = path or _db_path()
    init_verified_queries(path)
    vq_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _connect(path) as con:
        if source == "contributed":
            clause, params = _scope_clause(user_id, workspace_id)
            row = con.execute(
                f"SELECT COUNT(*) AS n FROM verified_queries WHERE source = 'contributed' AND {clause}",
                params,
            ).fetchone()
            if dict(row)["n"] >= CONTRIBUTED_CAP:
                raise ValueError(
                    f"Verified-query limit reached ({CONTRIBUTED_CAP}). "
                    "Keep these curated — remove one you no longer need first."
                )
        else:
            # Upsert per (creator, question, connection): keep the newest approval.
            con.execute(
                "DELETE FROM verified_queries"
                " WHERE source = 'gate' AND user_id = ? AND task = ? AND connection_id = ?",
                (user_id, task, connection_id),
            )

        con.execute(
            """
            INSERT INTO verified_queries
                (vq_id, source, user_id, workspace_id, name, task, sql,
                 task_embedding, connection_id, schema_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (vq_id, source, user_id, workspace_id, name.strip(), task.strip(),
             sql.strip(), _embed(task), connection_id, schema_hash, now),
        )
    return vq_id


def list_verified_queries(
    user_id: str,
    workspace_id: str = "",
    connection_id: str | None = None,
    path: str | None = None,
) -> list[dict[str, Any]]:
    path = path or _db_path()
    init_verified_queries(path)
    clause, params = _scope_clause(user_id, workspace_id)
    sql = (
        "SELECT vq_id, source, user_id, workspace_id, name, task, sql,"
        "       connection_id, schema_hash, created_at"
        f" FROM verified_queries WHERE {clause}"
    )
    if connection_id is not None:
        sql += " AND connection_id = ?"
        params = params + (connection_id,)
    sql += " ORDER BY created_at DESC"
    with _connect(path) as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def delete_verified_query(
    vq_id: str,
    user_id: str,
    workspace_id: str = "",
    path: str | None = None,
) -> bool:
    """Delete a row the caller can see. Returns True when a row was removed."""
    path = path or _db_path()
    init_verified_queries(path)
    clause, params = _scope_clause(user_id, workspace_id)
    with _connect(path) as con:
        before = dict(con.execute(
            f"SELECT COUNT(*) AS n FROM verified_queries WHERE vq_id = ? AND {clause}",
            (vq_id,) + params,
        ).fetchone())["n"]
        con.execute(
            f"DELETE FROM verified_queries WHERE vq_id = ? AND {clause}",
            (vq_id,) + params,
        )
    return before > 0


def retrieve_verified(
    task: str,
    *,
    user_id: str,
    workspace_id: str = "",
    connection_id: str = "",
    schema_hash: str = "",
    top_n: int = 2,
    min_similarity: float = 0.40,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Rank the caller's verified pairs against a new question.

    Returned dicts match the shape `_build_few_shot_block` expects
    (task, sql, similarity) plus source/name for callers that display them.
    """
    path = path or _db_path()
    init_verified_queries(path)

    try:
        from memory.semantic_cache import cosine_similarity, embed
        query_vec = embed(task)
    except Exception:  # noqa: BLE001 — no embedder, no retrieval
        return []

    clause, params = _scope_clause(user_id, workspace_id)
    with _connect(path) as con:
        rows = con.execute(
            "SELECT task, sql, task_embedding, source, name, connection_id, schema_hash"
            f" FROM verified_queries WHERE {clause}"
            " ORDER BY created_at DESC LIMIT 200",
            params,
        ).fetchall()

    scored: list[dict[str, Any]] = []
    for row in rows:
        try:
            d = dict(row)
            if not d.get("task_embedding"):
                continue
            vec = np.frombuffer(d["task_embedding"], dtype=np.float32)
            sim = cosine_similarity(query_vec, vec)
            sim += _SOURCE_BOOST.get(d["source"], 0.0)
            if connection_id and d.get("connection_id") == connection_id:
                sim += 0.05
            if schema_hash and d.get("schema_hash") and d["schema_hash"] != schema_hash:
                sim -= _SCHEMA_MISMATCH_PENALTY
            if sim < min_similarity:
                continue
            scored.append({
                "task": d["task"],
                "sql": d["sql"],
                "similarity": float(sim),
                "source": d["source"],
                "name": d.get("name") or "",
                "connection_id": d.get("connection_id") or "",
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("retrieve_verified: skipping malformed row: %s", exc)
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]
