"""
tests/test_security_fixes.py — Regression tests for security audit remediations.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory.store import init_db
from memory.semantic_cache import _ensure_cache_columns, check_cache, cosine_similarity
from tools.db_tools import DBConnection, validate_sql


class TestSqlGuards:
    def test_blocks_file_read_functions(self, tmp_duckdb):
        db = DBConnection("duckdb", path=tmp_duckdb)
        with pytest.raises(ValueError, match="File-read"):
            db.query("SELECT * FROM read_csv('/etc/passwd')")

    def test_blocks_copy(self, tmp_duckdb):
        db = DBConnection("duckdb", path=tmp_duckdb)
        with pytest.raises(ValueError, match="Mutation|privileged|SELECT/WITH"):
            db.query("COPY events TO '/tmp/out.csv'")

    def test_blocks_multi_statement(self, tmp_duckdb):
        with pytest.raises(ValueError, match="Multi-statement"):
            validate_sql("SELECT 1; DROP TABLE events")

    def test_appends_limit_when_missing(self, tmp_duckdb):
        db = DBConnection("duckdb", path=tmp_duckdb)
        df = db.query("SELECT user_id FROM events")
        assert len(df) <= 50000


class TestSemanticCacheUserIsolation:
    VEC = np.full(384, 1.0 / np.sqrt(384), dtype=np.float32)

    def _store(self, path: str, run_id: str, user_id: str, fingerprint: str, result: dict):
        init_db(path)
        _ensure_cache_columns(path)
        with sqlite3.connect(path) as con:
            con.execute(
                """INSERT OR IGNORE INTO runs (run_id, user_id, task, timestamp, analysis_mode)
                   VALUES (?, ?, 'task', datetime('now'), 'general')""",
                (run_id, user_id),
            )
            con.execute(
                """UPDATE runs
                      SET task_embedding = ?, cache_node_name = ?, cached_result = ?,
                          dataset_fingerprint = ?
                    WHERE run_id = ?""",
                (
                    self.VEC.tobytes(),
                    "generate_sql",
                    json.dumps(result),
                    fingerprint,
                    run_id,
                ),
            )

    def test_same_task_different_users_do_not_share_cache(self, tmp_path):
        path = str(tmp_path / "mem.db")
        run_a = str(uuid.uuid4())
        run_b = str(uuid.uuid4())
        fp = ""
        self._store(path, run_a, "user_a", fp, {"sql": "SELECT 1"})
        self._store(path, run_b, "user_b", fp, {"sql": "SELECT 2"})

        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """SELECT task_embedding, cached_result, user_id
                   FROM runs WHERE cache_node_name = ? AND dataset_fingerprint = ?""",
                ("generate_sql", fp),
            ).fetchall()

        best = None
        best_sim = 0.0
        for row in rows:
            if row["user_id"] != "user_a":
                continue
            vec = np.frombuffer(row["task_embedding"], dtype=np.float32)
            sim = cosine_similarity(self.VEC, vec)
            if sim > best_sim:
                best_sim = sim
                best = json.loads(row["cached_result"])

        assert best is not None
        assert best["sql"] == "SELECT 1"

        hit = check_cache("task", "generate_sql", dataset_fingerprint=fp, user_id="user_b", path=path)
        assert hit is None, "User B must not receive User A's cached SQL"
