"""Verified-query repository wiring: gate intake in log_run, retrieval in generate_sql."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory.semantic_cache as sc
from memory import verified_queries as vq


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    def _embed(text: str) -> np.ndarray:
        vec = np.zeros(384, dtype=np.float32)
        vec[hash(text.split()[0].lower()) % 384] = 1.0
        return vec
    monkeypatch.setattr(sc, "embed", _embed)


@pytest.fixture()
def mem_db(tmp_path, monkeypatch):
    path = str(tmp_path / "memory.db")
    monkeypatch.setenv("MEMORY_DB_PATH", path)
    return path


def _completed_state(**over):
    state = {
        "run_id": "run-1",
        "task": "revenue by week for the campaign",
        "user_id": "u1",
        "workspace_id": "ws1",
        "analysis_mode": "general",
        "generated_sql": "SELECT week, SUM(revenue) FROM revenue_weekly GROUP BY week",
        "query_approved": True,
        "narrative_draft": "Revenue held steady.",
        "schema_context": "TABLE: revenue_weekly\n  week INT\n  revenue DOUBLE",
    }
    state.update(over)
    return state


def _run_log_node(state, monkeypatch):
    import agents.analyze.nodes_narrative as nn

    monkeypatch.setattr(nn, "log_run", lambda **kw: None)
    monkeypatch.setattr(nn, "update_eval_score", lambda *a, **k: None)
    monkeypatch.setattr(nn, "flush", lambda: None)
    monkeypatch.setattr(nn, "_compute_quality_score", lambda s: 0.5)
    return nn.log_run_node(state)


class TestGateIntake:
    def test_completed_general_run_is_stored(self, mem_db, monkeypatch):
        _run_log_node(_completed_state(), monkeypatch)
        rows = vq.list_verified_queries("u1", workspace_id="ws1", path=mem_db)
        assert len(rows) == 1
        assert rows[0]["source"] == "gate"
        assert "revenue_weekly" in rows[0]["sql"]
        assert rows[0]["schema_hash"]  # stamped for later invalidation

    def test_unapproved_sql_is_not_stored(self, mem_db, monkeypatch):
        _run_log_node(_completed_state(query_approved=False), monkeypatch)
        assert vq.list_verified_queries("u1", workspace_id="ws1", path=mem_db) == []

    def test_guest_runs_are_not_stored(self, mem_db, monkeypatch):
        _run_log_node(_completed_state(user_id="guest-abc123"), monkeypatch)
        assert vq.list_verified_queries("guest-abc123", path=mem_db) == []

    def test_incomplete_run_without_narrative_is_not_stored(self, mem_db, monkeypatch):
        _run_log_node(_completed_state(narrative_draft="", final_narrative=""), monkeypatch)
        assert vq.list_verified_queries("u1", workspace_id="ws1", path=mem_db) == []

    def test_store_failure_never_breaks_the_run(self, mem_db, monkeypatch):
        import agents.analyze.nodes_narrative as nn
        from memory import verified_queries as vq_mod

        monkeypatch.setattr(vq_mod, "add_verified_query",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
        out = _run_log_node(_completed_state(), monkeypatch)
        assert out["run_id"] == "run-1"


class TestRetrievalInPrompt:
    def _fake_client(self, captured):
        class Messages:
            def create(self, **kw):
                captured.update(kw)
                block = SimpleNamespace(type="text", text="```sql\nSELECT 1\n```")
                return SimpleNamespace(
                    content=[block], stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                          cache_read_input_tokens=0,
                                          cache_creation_input_tokens=0),
                )
        return SimpleNamespace(messages=Messages())

    def test_verified_query_reaches_the_prompt(self, mem_db, monkeypatch):
        import agents.analyze.nodes_sql as ns

        vq.add_verified_query(
            "revenue by week, canonical", "SELECT week, SUM(revenue) FROM revenue_weekly GROUP BY week",
            source="contributed", user_id="u1", workspace_id="ws1",
            name="Canonical weekly revenue", path=mem_db,
        )
        captured: dict = {}
        monkeypatch.setattr(ns, "_anthropic_client", lambda: self._fake_client(captured))

        out = ns.generate_sql({
            "task": "revenue for the last month",
            "analysis_mode": "general",
            "db_backend": "duckdb",
            "schema_context": "TABLE: revenue_weekly\n  week INT\n  revenue DOUBLE",
            "relevant_history": [],
            "user_id": "u1",
            "workspace_id": "ws1",
        })

        assert out["generated_sql"] == "SELECT 1"
        prompt_text = str(captured.get("messages"))
        assert "Canonical weekly revenue" in prompt_text or "revenue_weekly GROUP BY week" in prompt_text

    def test_stale_table_examples_still_filtered(self, mem_db, monkeypatch):
        """A verified pair whose tables vanished must not reach the prompt —
        the existing schema filter is the last line of defence."""
        import agents.analyze.nodes_sql as ns

        vq.add_verified_query(
            "revenue by week, canonical", "SELECT * FROM dropped_table",
            source="contributed", user_id="u1", path=mem_db,
        )
        captured: dict = {}
        monkeypatch.setattr(ns, "_anthropic_client", lambda: self._fake_client(captured))

        ns.generate_sql({
            "task": "revenue for the last month",
            "analysis_mode": "general",
            "db_backend": "duckdb",
            "schema_context": "TABLE: revenue_weekly\n  week INT\n  revenue DOUBLE",
            "relevant_history": [],
            "user_id": "u1",
        })
        assert "dropped_table" not in str(captured.get("messages"))
