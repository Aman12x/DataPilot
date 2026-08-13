"""
tests/test_log_run_node.py — log_run_node end-to-end with an A/B-shaped result.

A June 2026 refactor left log_run_node passing user_id= to a store_cache that
never accepted it. The TypeError killed every A/B run at the final graph node
for six weeks, and the semantic cache stored nothing in that window — CI never
noticed because no test executed log_run_node with a result that had the
required A/B columns (that branch is the only path into store_cache).

This test runs the real node with the real store against a temp DB; only the
embedding model is stubbed. If the node/store signatures ever drift again,
this fails with the same TypeError production would see.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.analysis_config import load_metric_config


@pytest.fixture
def mem_db(tmp_path, monkeypatch):
    path = str(tmp_path / "mem.db")
    monkeypatch.setenv("MEMORY_DB_PATH", path)
    # embed() downloads MiniLM — stub it; the cache row just needs bytes.
    import memory.semantic_cache as sc
    monkeypatch.setattr(sc, "embed", lambda text: np.zeros(384, dtype=np.float32))
    return path


def test_ab_run_reaches_semantic_cache_store(mem_db):
    from agents.analyze.nodes_narrative import log_run_node
    from memory.store import get_run

    mc = load_metric_config()
    df = pd.DataFrame({
        mc.primary_metric: [1.0, 2.0],
        mc.covariate:      [0.5, 0.6],
        "variant":         ["control", "treatment"],
    })
    out = log_run_node({
        "run_id":          "test-ab-run",
        "task":            "did the experiment work?",
        "analysis_mode":   "ab_test",
        "user_id":         "analyst-1",
        "query_type":      "exploratory",
        "generated_sql":   "SELECT 1",
        "query_approved":  True,
        "query_result":    df,
        "narrative_draft": "It worked.",
        "final_narrative": "It worked.",
        "metric_config":   mc,
    })
    assert out["run_id"] == "test-ab-run"

    row = get_run("test-ab-run", path=mem_db)
    assert row is not None
    assert row["user_id"] == "analyst-1"
    # The store_cache branch actually executed: cache columns are populated.
    assert row["cache_node_name"] == "generate_sql"
    assert row["cached_result"]


def test_lookup_run_logs_without_cache_branch(mem_db):
    from agents.analyze.nodes_narrative import log_run_node
    from memory.store import get_run

    out = log_run_node({
        "run_id":          "test-lookup-run",
        "task":            "what was dau?",
        "analysis_mode":   "general",
        "query_type":      "lookup",
        "user_id":         "analyst-1",
        "query_result":    pd.DataFrame({"dau": [1]}),
        "narrative_draft": "**dau**: 1",
        "final_narrative": "**dau**: 1",
    })
    assert out["run_id"] == "test-lookup-run"
    row = get_run("test-lookup-run", path=mem_db)
    assert row["query_type"] == "lookup"
