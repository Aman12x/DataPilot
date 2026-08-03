"""Verified-query repository: store, scope, cap, upsert, retrieval ranking."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory.semantic_cache as sc
from memory import verified_queries as vq


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    """Deterministic embedder: same first word → identical vector, different
    first word → orthogonal. Keeps tests fast and model-free."""
    def _embed(text: str) -> np.ndarray:
        vec = np.zeros(384, dtype=np.float32)
        vec[hash(text.split()[0].lower()) % 384] = 1.0
        return vec
    monkeypatch.setattr(sc, "embed", _embed)


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "vq.db")


def test_add_and_list_personal_scope(db):
    vq.add_verified_query("revenue by week", "SELECT 1", source="gate",
                          user_id="u1", path=db)
    assert len(vq.list_verified_queries("u1", path=db)) == 1
    # Another user sees nothing.
    assert vq.list_verified_queries("u2", path=db) == []


def test_workspace_rows_visible_to_members(db):
    vq.add_verified_query("revenue by week", "SELECT 1", source="contributed",
                          user_id="owner", workspace_id="ws1", name="Weekly revenue", path=db)
    # A different member of the same workspace sees the row.
    assert len(vq.list_verified_queries("analyst", workspace_id="ws1", path=db)) == 1
    # A member of another workspace does not.
    assert vq.list_verified_queries("outsider", workspace_id="ws2", path=db) == []


def test_gate_intake_upserts_per_question_and_connection(db):
    vq.add_verified_query("revenue by week", "SELECT 1", source="gate",
                          user_id="u1", connection_id="c1", path=db)
    vq.add_verified_query("revenue by week", "SELECT 2 -- newer", source="gate",
                          user_id="u1", connection_id="c1", path=db)
    rows = vq.list_verified_queries("u1", path=db)
    assert len(rows) == 1
    assert rows[0]["sql"] == "SELECT 2 -- newer"
    # A different connection is a separate pair.
    vq.add_verified_query("revenue by week", "SELECT 3", source="gate",
                          user_id="u1", connection_id="c2", path=db)
    assert len(vq.list_verified_queries("u1", path=db)) == 2


def test_contributed_cap_forces_curation(db, monkeypatch):
    monkeypatch.setattr(vq, "CONTRIBUTED_CAP", 2)
    vq.add_verified_query("q1 a", "SELECT 1", source="contributed", user_id="u1", path=db)
    vq.add_verified_query("q2 b", "SELECT 2", source="contributed", user_id="u1", path=db)
    with pytest.raises(ValueError, match="limit"):
        vq.add_verified_query("q3 c", "SELECT 3", source="contributed", user_id="u1", path=db)
    # Gate intake is not capped.
    vq.add_verified_query("q4 d", "SELECT 4", source="gate", user_id="u1", path=db)


def test_delete_respects_scope(db):
    vq_id = vq.add_verified_query("revenue by week", "SELECT 1", source="contributed",
                                  user_id="u1", path=db)
    assert vq.delete_verified_query(vq_id, "someone-else", path=db) is False
    assert len(vq.list_verified_queries("u1", path=db)) == 1
    assert vq.delete_verified_query(vq_id, "u1", path=db) is True
    assert vq.list_verified_queries("u1", path=db) == []


def test_retrieval_ranks_contributed_above_gate(db):
    vq.add_verified_query("revenue by week", "SELECT 'gate'", source="gate",
                          user_id="u1", path=db)
    vq.add_verified_query("revenue by month", "SELECT 'contributed'", source="contributed",
                          user_id="u1", name="Canonical revenue", path=db)
    got = vq.retrieve_verified("revenue please", user_id="u1", top_n=2, path=db)
    assert [g["source"] for g in got] == ["contributed", "gate"]


def test_retrieval_filters_dissimilar_questions(db):
    vq.add_verified_query("churn cohort", "SELECT 1", source="gate", user_id="u1", path=db)
    assert vq.retrieve_verified("revenue by week", user_id="u1", path=db) == []


def test_schema_mismatch_demotes(db):
    vq.add_verified_query("revenue old-schema", "SELECT 'stale'", source="gate",
                          user_id="u1", schema_hash="old", path=db)
    vq.add_verified_query("revenue new-schema", "SELECT 'fresh'", source="gate",
                          user_id="u1", schema_hash="new", path=db)
    got = vq.retrieve_verified("revenue now", user_id="u1",
                               schema_hash="new", top_n=2, path=db)
    assert got and got[0]["sql"] == "SELECT 'fresh'"


def test_rejects_blank_input(db):
    with pytest.raises(ValueError):
        vq.add_verified_query("  ", "SELECT 1", source="gate", user_id="u1", path=db)
    with pytest.raises(ValueError):
        vq.add_verified_query("task", "", source="contributed", user_id="u1", path=db)
    with pytest.raises(ValueError, match="source"):
        vq.add_verified_query("task", "SELECT 1", source="mystery", user_id="u1", path=db)
