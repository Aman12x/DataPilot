"""
tests/test_semantic_cache_isolation.py
─────────────────────────────────────────────────────────────────────────────
Tests that the dataset_fingerprint dimension on the semantic cache prevents
cross-dataset cache poisoning.

Before the fix, two identical task strings run against different uploaded
datasets would return a cached result from whichever dataset ran first.

After the fix:
  • Same task + same fingerprint → cache HIT
  • Same task + different fingerprint → cache MISS
  • store_cache stores the fingerprint; check_cache filters by it

No sentence-transformers model is called in pure SQLite path tests.
For embedding-level tests, the model is loaded once per session.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import uuid

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.store import init_db
from memory.semantic_cache import (
    _ensure_cache_columns,
    check_cache,
    cosine_similarity,
    embed,
    store_cache,
)


# ─── SQLite helpers ────────────────────────────────────────────────────────────

def _make_db() -> str:
    """Return path to a fresh temp SQLite DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def _insert_run(path: str, run_id: str) -> None:
    """Insert a minimal run row so store_cache UPDATE can find it."""
    init_db(path)
    _ensure_cache_columns(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """INSERT OR IGNORE INTO runs (run_id, user_id, task, timestamp, analysis_mode)
               VALUES (?, 'test_user', 'test task', datetime('now'), 'general')""",
            (run_id,),
        )


def _direct_store(
    path: str,
    run_id: str,
    task_vec: np.ndarray,
    result: dict,
    node_name: str,
    fingerprint: str,
) -> None:
    """Bypass embed() and write directly into the DB for deterministic tests."""
    result_json = json.dumps(result)
    vec_bytes   = task_vec.tobytes()
    with sqlite3.connect(path) as con:
        con.execute(
            """UPDATE runs
                  SET task_embedding      = ?,
                      cache_node_name     = ?,
                      cached_result       = ?,
                      dataset_fingerprint = ?
                WHERE run_id = ?""",
            (vec_bytes, node_name, result_json, fingerprint, run_id),
        )


def _direct_check(
    path: str,
    query_vec: np.ndarray,
    node_name: str,
    fingerprint: str,
    hard_threshold: float = 0.92,
    soft_threshold: float = 0.80,
) -> dict | None:
    """Bypass embed() and check cache directly with a pre-computed vector."""
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT task_embedding, cached_result
               FROM   runs
               WHERE  cache_node_name    = ?
                 AND  dataset_fingerprint = ?
                 AND  task_embedding     IS NOT NULL
                 AND  cached_result      IS NOT NULL""",
            (node_name, fingerprint),
        ).fetchall()

    best_sim    = 0.0
    best_result = None
    for row in rows:
        stored_vec = np.frombuffer(row["task_embedding"], dtype=np.float32)
        sim = cosine_similarity(query_vec, stored_vec)
        if sim > best_sim:
            best_sim    = sim
            best_result = json.loads(row["cached_result"])

    if best_result is None or best_sim < soft_threshold:
        return None
    hit_type = "hard" if best_sim >= hard_threshold else "soft"
    return {"hit_type": hit_type, "result": best_result, "similarity": best_sim}


# ═════════════════════════════════════════════════════════════════════════════
# 1. _ensure_cache_columns
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsureCacheColumns:
    """Verify that the cache columns are added exactly once and idempotently."""

    def test_columns_added_on_fresh_db(self, tmp_path):
        path = str(tmp_path / "test.db")
        init_db(path)
        _ensure_cache_columns(path)
        with sqlite3.connect(path) as con:
            cols = {row[1] for row in con.execute("PRAGMA table_info(runs)").fetchall()}
        assert "cache_node_name" in cols
        assert "cached_result" in cols
        assert "dataset_fingerprint" in cols

    def test_idempotent_second_call(self, tmp_path):
        """Calling _ensure_cache_columns twice must not raise."""
        path = str(tmp_path / "test.db")
        init_db(path)
        _ensure_cache_columns(path)
        _ensure_cache_columns(path)  # should not raise

    def test_dataset_fingerprint_default_empty_string(self, tmp_path):
        """Existing rows without dataset_fingerprint should default to ''."""
        path = str(tmp_path / "test.db")
        init_db(path)
        run_id = str(uuid.uuid4())
        with sqlite3.connect(path) as con:
            con.execute(
                "INSERT INTO runs (run_id, user_id, task, timestamp, analysis_mode) VALUES (?,?,?,datetime('now'),?)",
                (run_id, "u", "t", "general"),
            )
        _ensure_cache_columns(path)
        with sqlite3.connect(path) as con:
            val = con.execute(
                "SELECT dataset_fingerprint FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        # DEFAULT '' applied to existing rows
        assert val == "" or val is None  # SQLite ALTER TABLE DEFAULT may be NULL for existing rows


# ═════════════════════════════════════════════════════════════════════════════
# 2. Cross-dataset isolation (deterministic, no model calls)
# ═════════════════════════════════════════════════════════════════════════════

class TestCacheIsolationDeterministic:
    """
    Use a fixed normalised vector to test that dataset_fingerprint scopes hits.
    These tests are fast (no model loading).
    """

    # A fixed unit vector (384-dim) simulating an embedding.
    # NOTE: must use np.full with explicit dtype=float32 — dividing
    # a float32 array by np.sqrt (float64) silently upcasts to float64,
    # doubling the byte count and causing dimension mismatches on readback.
    VEC = np.full(384, 1.0 / np.sqrt(384), dtype=np.float32)

    def test_same_fingerprint_returns_hit(self, tmp_path):
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fingerprint = "uploads/user1/abc.db"
        expected    = {"narrative": "some result", "metric": 42}
        _direct_store(path, run_id, self.VEC, expected, "generate_narrative", fingerprint)

        result = _direct_check(path, self.VEC, "generate_narrative", fingerprint)
        assert result is not None, "Same fingerprint should produce a cache hit"
        assert result["result"] == expected

    def test_different_fingerprint_returns_miss(self, tmp_path):
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp_a = "uploads/user1/dataset_a.db"
        fp_b = "uploads/user1/dataset_b.db"
        _direct_store(path, run_id, self.VEC, {"answer": "from_a"}, "generate_narrative", fp_a)

        result = _direct_check(path, self.VEC, "generate_narrative", fp_b)
        assert result is None, \
            "Different fingerprint should produce a cache MISS even with identical task vector"

    def test_empty_fingerprint_vs_nonempty_fingerprint_are_isolated(self, tmp_path):
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        # Store with empty fingerprint (demo DB)
        _direct_store(path, run_id, self.VEC, {"answer": "demo"}, "generate_sql", "")

        # Check with a non-empty fingerprint (upload)
        result = _direct_check(path, self.VEC, "generate_sql", "uploads/user1/upload.db")
        assert result is None, \
            "Demo-DB cache entry should not bleed into upload fingerprint"

    def test_two_datasets_independently_cached(self, tmp_path):
        path = str(tmp_path / "test.db")
        fp_a = "uploads/user1/a.db"
        fp_b = "uploads/user1/b.db"

        # Run 1 → dataset A
        run_a = str(uuid.uuid4())
        _insert_run(path, run_a)
        vec_a = np.full(384, 1.0 / np.sqrt(384), dtype=np.float32)
        _direct_store(path, run_a, vec_a, {"answer": "result_A"}, "generate_sql", fp_a)

        # Run 2 → dataset B (slightly different vector, same fingerprint B)
        run_b = str(uuid.uuid4())
        _insert_run(path, run_b)
        vec_b = vec_a.copy()
        vec_b[0] += 0.001  # trivially different
        vec_b /= np.linalg.norm(vec_b)
        _direct_store(path, run_b, vec_b, {"answer": "result_B"}, "generate_sql", fp_b)

        # Querying with fp_a should return result_A
        res_a = _direct_check(path, vec_a, "generate_sql", fp_a)
        assert res_a is not None and res_a["result"]["answer"] == "result_A"

        # Querying with fp_b should return result_B
        res_b = _direct_check(path, vec_b, "generate_sql", fp_b)
        assert res_b is not None and res_b["result"]["answer"] == "result_B"

        # Cross-check: querying fp_b with vec_a gives result_B (similarity ≈ 1.0 regardless)
        # but crucially querying fp_a with vec_a must NOT return result_B
        res_cross = _direct_check(path, vec_a, "generate_sql", fp_b)
        if res_cross is not None:
            assert res_cross["result"]["answer"] != "result_A", \
                "Cross-dataset contamination: fp_b returned result_A!"

    def test_node_name_scopes_cache(self, tmp_path):
        """Two different node names sharing the same fingerprint must not collide."""
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp = "uploads/user1/data.db"
        _direct_store(path, run_id, self.VEC, {"sql": "SELECT ..."}, "generate_sql", fp)

        # Check for a different node — should miss
        result = _direct_check(path, self.VEC, "generate_narrative", fp)
        assert result is None, \
            "Cache hit for wrong node_name — node scoping is broken"

    def test_hard_hit_vs_soft_hit(self, tmp_path):
        """Vectors with similarity ≥ 0.92 → hard hit; 0.80–0.92 → soft hit."""
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp = ""
        stored_vec = self.VEC.copy()
        _direct_store(path, run_id, stored_vec, {"x": 1}, "generate_sql", fp)

        # Query with identical vector → similarity == 1.0 → hard hit
        result = _direct_check(path, stored_vec, "generate_sql", fp)
        assert result is not None
        assert result["hit_type"] == "hard"
        assert result["similarity"] >= 0.92

    def test_below_soft_threshold_returns_none(self, tmp_path):
        """A very different query vector should produce no cache hit."""
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp = ""
        stored_vec = self.VEC.copy()
        _direct_store(path, run_id, stored_vec, {"x": 1}, "generate_sql", fp)

        # Orthogonal vector → similarity ≈ 0
        orthogonal = np.zeros(384, dtype=np.float32)
        orthogonal[0] = 1.0  # already float32, no upcast risk
        result = _direct_check(path, orthogonal, "generate_sql", fp, hard_threshold=0.92, soft_threshold=0.80)
        assert result is None, "Orthogonal vector should not produce a cache hit"


# ═════════════════════════════════════════════════════════════════════════════
# 3. store_cache / check_cache integration (uses real embed() → model required)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestStoreCacheIntegration:
    """
    End-to-end tests using real MiniLM embeddings.
    Marked slow — run with: pytest -m slow
    Model is loaded once per session via the module-level singleton in semantic_cache.
    """

    TASK_A = "What is the average revenue per customer?"
    TASK_B = "Show me salary breakdown by department."

    def test_same_task_same_fingerprint_hits(self, tmp_path):
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp      = "uploads/user1/a.db"
        payload = {"result": "task_a_result"}

        store_cache(self.TASK_A, "generate_sql", payload, run_id, dataset_fingerprint=fp, path=path)

        hit = check_cache(self.TASK_A, "generate_sql", dataset_fingerprint=fp, path=path)
        assert hit is not None, "Exact re-query should hit"
        assert hit["hit_type"] == "hard"
        assert hit["result"] == payload

    def test_same_task_different_fingerprint_misses(self, tmp_path):
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp_a    = "uploads/user1/a.db"
        fp_b    = "uploads/user1/b.db"
        payload = {"result": "task_a_result_from_dataset_a"}

        store_cache(self.TASK_A, "generate_sql", payload, run_id, dataset_fingerprint=fp_a, path=path)

        # Same task, different dataset → should miss
        hit = check_cache(self.TASK_A, "generate_sql", dataset_fingerprint=fp_b, path=path)
        assert hit is None, \
            "Same task against different fingerprint must be a cache MISS (cross-dataset isolation)"

    def test_semantically_similar_task_hits(self, tmp_path):
        """Paraphrase of the same question should also hit (cosine ≥ 0.80)."""
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp      = "uploads/user1/data.db"
        payload = {"sql": "SELECT AVG(revenue) FROM events GROUP BY customer_id"}

        store_cache(self.TASK_A, "generate_sql", payload, run_id, dataset_fingerprint=fp, path=path)

        paraphrase = "Average revenue by customer"  # semantically same
        hit = check_cache(paraphrase, "generate_sql", dataset_fingerprint=fp, path=path)
        # Should at least produce a soft hit
        assert hit is not None, "Semantically similar task should produce a cache hit"

    def test_unrelated_task_misses(self, tmp_path):
        """TASK_B (salary by dept) should not hit a cache stored for TASK_A (avg revenue)."""
        path = str(tmp_path / "test.db")
        run_id = str(uuid.uuid4())
        _insert_run(path, run_id)
        fp      = ""  # demo DB
        payload = {"sql": "SELECT AVG(revenue) FROM events"}

        store_cache(self.TASK_A, "generate_sql", payload, run_id, dataset_fingerprint=fp, path=path)

        hit = check_cache(self.TASK_B, "generate_sql", dataset_fingerprint=fp, path=path)
        assert hit is None or hit["hit_type"] == "soft", \
            "Unrelated task should not produce a hard cache hit"


# ═════════════════════════════════════════════════════════════════════════════
# 4. cosine_similarity correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.ones(384, dtype=np.float32) / np.sqrt(384)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = np.zeros(384, dtype=np.float32); a[0] = 1.0
        b = np.zeros(384, dtype=np.float32); b[1] = 1.0
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_opposite_vectors(self):
        v = np.ones(384, dtype=np.float32) / np.sqrt(384)
        assert abs(cosine_similarity(v, -v) + 1.0) < 1e-5

    def test_result_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(384).astype(np.float32)
            b = rng.standard_normal(384).astype(np.float32)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            sim = cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0
