"""
tests/test_memory_store.py — Unit tests for memory/store.py, retriever.py,
and semantic_cache.py.

All tests use tmp_path — never touch the real MEMORY_DB_PATH.
"""

import os
import numpy as np
import pytest

from memory.store import init_db, log_run, get_run, get_all_runs
from memory.retriever import retrieve_relevant_history
from memory.semantic_cache import embed, cosine_similarity, check_cache, store_cache


# ── store.py ──────────────────────────────────────────────────────────────────

def test_log_and_retrieve_run(tmp_path):
    """log_run persists a run; get_run returns it with matching fields."""
    db = str(tmp_path / "test.db")
    run_id = log_run(
        "why did DAU drop on android?",
        path=db,
        metric="dau_rate",
        top_segment="platform=android,user_segment=new",
        eval_score=0.9,
    )
    result = get_run(run_id, path=db)
    assert result is not None
    assert result["run_id"] == run_id
    assert result["metric"] == "dau_rate"
    assert result["eval_score"] == 0.9


def test_analyst_override_roundtrips_as_dict(tmp_path):
    """analyst_override is stored as JSON and returned as a dict."""
    db = str(tmp_path / "test.db")
    override = {"covariate": "session_count", "reason": "analyst preferred"}
    run_id = log_run("investigate retention drop", path=db, analyst_override=override)
    result = get_run(run_id, path=db)
    assert result["analyst_override"] == override


def test_get_all_runs_ordered_by_recency(tmp_path):
    """get_all_runs returns most recent run first."""
    db = str(tmp_path / "test.db")
    id1 = log_run("first task",  path=db)
    id2 = log_run("second task", path=db)
    runs = get_all_runs(path=db)
    assert runs[0]["run_id"] == id2
    assert runs[1]["run_id"] == id1


# ── retriever.py ──────────────────────────────────────────────────────────────

def test_retriever_returns_relevant_run(tmp_path):
    """retrieve_relevant_history returns runs with keyword overlap (audit_passed only)."""
    db = str(tmp_path / "test.db")
    log_run("DAU drop investigation android new users", path=db,
            metric="dau_rate", top_segment="platform=android,user_segment=new",
            audit_passed=True)
    log_run("revenue analysis Q4 unrelated", path=db, metric="revenue", audit_passed=True)

    results = retrieve_relevant_history("why did DAU drop for android?", path=db)
    assert len(results) >= 1
    assert "android" in results[0]["top_segment"]


def test_retriever_returns_empty_for_no_overlap(tmp_path):
    """retrieve_relevant_history returns [] when nothing overlaps."""
    db = str(tmp_path / "test.db")
    log_run("revenue Q4 analysis", path=db, metric="revenue")

    results = retrieve_relevant_history("funnel drop-off iOS install step", path=db)
    assert results == []


# ── semantic_cache.py ─────────────────────────────────────────────────────────

def test_embed_returns_unit_vector():
    """embed() returns a float32 vector with unit norm."""
    vec = embed("why did DAU drop on android?")
    assert vec.dtype == np.float32
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_cosine_similarity_identical():
    """Identical vectors have cosine similarity = 1.0."""
    vec = embed("investigate DAU drop")
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-5


def test_cosine_similarity_different():
    """Unrelated sentences have similarity well below 1.0."""
    v1 = embed("DAU drop android new users experiment")
    v2 = embed("quarterly revenue forecast model")
    assert cosine_similarity(v1, v2) < 0.8


def test_cache_miss_on_empty_db(tmp_path):
    """check_cache returns None when the DB has no stored embeddings."""
    db = str(tmp_path / "test.db")
    result = check_cache("why did DAU drop?", "generate_sql", path=db)
    assert result is None


def test_cache_hard_hit(tmp_path, monkeypatch):
    """A near-identical task returns a hard cache hit."""
    db = str(tmp_path / "test.db")

    # Lower threshold so the test doesn't depend on exact MiniLM similarity
    monkeypatch.setenv("SEMANTIC_CACHE_HARD_THRESHOLD", "0.80")
    monkeypatch.setenv("SEMANTIC_CACHE_SOFT_THRESHOLD", "0.50")

    task = "why did DAU drop on android?"
    run_id = log_run(task, path=db)
    store_cache(task, "generate_sql", {"sql": "SELECT 1"}, run_id, path=db)

    # Same task → should be a hard hit
    hit = check_cache(task, "generate_sql", path=db)
    assert hit is not None
    assert hit["hit_type"] == "hard"
    assert hit["result"] == {"sql": "SELECT 1"}
