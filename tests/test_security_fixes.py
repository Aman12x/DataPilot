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
from tools import db_tools
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

    def test_semicolon_inside_comment_is_not_multi_statement(self):
        """The LLM annotates SQL with `-- Assumption: …` comments; a semicolon
        in the prose must not fail the multi-statement check."""
        validate_sql(
            '-- Assumption: "completed" means finished (completed = 1); counting users\n'
            "SELECT step, COUNT(*) FROM funnel GROUP BY step"
        )

    def test_semicolon_inside_string_literal_is_not_multi_statement(self):
        validate_sql("SELECT * FROM events WHERE user_segment = 'a;b'")

    def test_mutation_keyword_inside_literal_or_comment_is_allowed(self):
        validate_sql("SELECT 'DROP-off rate' AS label FROM events  -- delete nothing")

    def test_comment_does_not_hide_a_real_second_statement(self):
        with pytest.raises(ValueError, match="Multi-statement"):
            validate_sql("SELECT 1 -- note\n; DROP TABLE events")

    def test_double_dash_inside_literal_does_not_start_comment(self):
        """A literal containing `--` must not comment out the rest of the
        statement, or a trailing `; DROP …` would slip past the check."""
        with pytest.raises(ValueError, match="Multi-statement"):
            validate_sql("SELECT '--' AS c; DROP TABLE events")

    def test_never_truncates_a_result(self, tmp_duckdb):
        """No implicit LIMIT: an unlimited SELECT returns every row.

        The old behaviour appended `LIMIT 50000`, which handed the stats tools a
        non-deterministic subset (no ORDER BY) and reported the effect size as
        if it covered the population.
        """
        db = DBConnection("duckdb", path=tmp_duckdb)
        df = db.query("SELECT user_id FROM events")
        assert len(df) == 3
        assert set(df["user_id"]) == {"u1", "u2", "u3"}

    def test_oversized_result_raises_instead_of_truncating(self, tmp_duckdb, monkeypatch):
        """Above the ceiling the query fails loudly — it never returns a prefix."""
        monkeypatch.setattr(db_tools, "_MAX_MATERIALIZE_ROWS", 2)
        db = DBConnection("duckdb", path=tmp_duckdb)
        with pytest.raises(db_tools.ResultTooLargeError) as exc:
            db.query("SELECT user_id FROM events")
        assert exc.value.rows == 3
        assert "Aggregate in SQL" in str(exc.value)

    def test_row_budget_can_be_disabled(self, tmp_duckdb, monkeypatch):
        monkeypatch.setattr(db_tools, "_MAX_MATERIALIZE_ROWS", 0)
        db = DBConnection("duckdb", path=tmp_duckdb)
        assert len(db.query("SELECT user_id FROM events")) == 3

    def test_precount_survives_a_query_it_cannot_wrap(self, tmp_duckdb, monkeypatch):
        """A precount failure must not block the real query — it is a guard rail."""
        monkeypatch.setattr(db_tools, "_MAX_MATERIALIZE_ROWS", 1)
        monkeypatch.setattr(
            db_tools, "_count_wrapper", lambda sql: "SELECT COUNT(*) FROM _no_such_table"
        )
        db = DBConnection("duckdb", path=tmp_duckdb)
        assert len(db.query("SELECT user_id FROM events")) == 3


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
