"""
tests/test_lookup_fastpath.py — the lookup fast path.

A general-mode question classified "lookup" with a small result skips the
analysis gate, the narrative LLM call, the audit, and the narrative gate:
execute_query → describe_data → generate_charts → generate_narrative
(deterministic render) → log_run. Everything here defends two properties:

1. The fast path never fires for A/B tests, power analyses, or any result
   large enough to suggest the classifier was wrong (self-healing).
2. Every routing decision and the narrative node share one predicate
   (is_fast_lookup), so the branches cannot disagree.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.analyze import node_shared
from agents.analyze.node_shared import is_fast_lookup
from agents.analyze.graph import (
    _route_after_describe_data,
    _route_after_generate_charts,
    _route_after_generate_narrative,
)
from agents.analyze.nodes_narrative import (
    _render_lookup_answer,
    generate_narrative,
)


def _lookup_state(rows: int = 1, **overrides) -> dict:
    state = {
        "analysis_mode": "general",
        "query_type":    "lookup",
        "query_result":  pd.DataFrame({"dau": range(100, 100 + rows)}),
        "task":          "what was DAU yesterday?",
    }
    state.update(overrides)
    return state


class TestIsFastLookup:
    def test_small_general_lookup_qualifies(self):
        assert is_fast_lookup(_lookup_state())

    def test_ab_test_mode_never_qualifies(self):
        assert not is_fast_lookup(_lookup_state(analysis_mode="ab_test"))

    def test_exploratory_never_qualifies(self):
        assert not is_fast_lookup(_lookup_state(query_type="exploratory"))

    def test_oversized_result_disqualifies(self):
        """Row count above the cap means the classifier was wrong — full pipeline."""
        assert not is_fast_lookup(_lookup_state(rows=node_shared._LOOKUP_MAX_ROWS + 1))

    def test_at_the_cap_still_qualifies(self):
        assert is_fast_lookup(_lookup_state(rows=node_shared._LOOKUP_MAX_ROWS))

    def test_empty_result_disqualifies(self):
        assert not is_fast_lookup(_lookup_state(query_result=pd.DataFrame()))

    def test_missing_result_disqualifies(self):
        assert not is_fast_lookup(_lookup_state(query_result=None))


class TestRouting:
    def test_fast_lookup_skips_correlations(self):
        assert _route_after_describe_data(_lookup_state()) == "generate_charts"

    def test_misclassified_big_lookup_runs_full_pipeline(self):
        state = _lookup_state(rows=node_shared._LOOKUP_MAX_ROWS + 1)
        assert _route_after_describe_data(state) == "find_correlations"

    def test_fast_lookup_skips_analysis_gate(self):
        assert _route_after_generate_charts(_lookup_state()) == "generate_narrative"

    def test_exploratory_still_hits_analysis_gate(self):
        state = _lookup_state(query_type="exploratory")
        assert _route_after_generate_charts(state) == "analysis_gate"

    def test_ab_test_still_hits_analysis_gate(self):
        assert _route_after_generate_charts(_lookup_state(analysis_mode="ab_test")) == "analysis_gate"

    def test_fast_lookup_skips_narrative_gate(self):
        assert _route_after_generate_narrative(_lookup_state()) == "log_run"

    def test_exploratory_still_hits_narrative_gate(self):
        state = _lookup_state(query_type="exploratory")
        assert _route_after_generate_narrative(state) == "narrative_gate"

    def test_audit_block_no_longer_regenerates(self):
        """Patch-only: audit findings are fixed in place inside the node; a
        blocked audit goes to the gate as a warning, never back around the
        88s narrative+audit loop."""
        state = _lookup_state(
            query_type="exploratory",
            audit_blocked=True,
            narrative_revision_count=1,
        )
        assert _route_after_generate_narrative(state) == "narrative_gate"


class TestDeterministicNarrative:
    def test_no_llm_call_on_fast_path(self, monkeypatch):
        """The whole point: a fast lookup must never touch the Anthropic client."""
        def _boom():
            raise AssertionError("fast lookup made an LLM call")
        # nodes_narrative copies node_shared's globals at import
        # (globals().update(vars(_shared))), so patch ITS reference — patching
        # node_shared would leave the call site holding the real client.
        import agents.analyze.nodes_narrative as nn
        monkeypatch.setattr(nn, "_anthropic_client", _boom)

        out = generate_narrative(_lookup_state())
        assert out["final_narrative"]
        assert out["narrative_approved"] is True
        assert out["audit_result"] is None
        # No cost keys returned — nothing was spent
        assert "estimated_cost_usd" not in out

    def test_scalar_renders_as_value(self):
        df = pd.DataFrame({"dau": [12345]})
        assert _render_lookup_answer(df) == "**dau**: 12,345"

    def test_single_row_renders_as_field_list(self):
        df = pd.DataFrame({"dau": [12345], "platform": ["ios"]})
        out = _render_lookup_answer(df)
        assert "- **dau**: 12,345" in out
        assert "- **platform**: ios" in out

    def test_small_table_renders_as_markdown(self):
        df = pd.DataFrame({"product": ["a|b", "c"], "revenue": [1000.5, 2000.0]})
        out = _render_lookup_answer(df)
        assert out.splitlines()[0] == "| product | revenue |"
        assert "a\\|b" in out              # pipes escaped, table stays intact
        assert "```" not in out            # sanitiseNarrative strips fences

    def test_null_value_renders_as_dash(self):
        df = pd.DataFrame({"dau": [None]})
        assert _render_lookup_answer(df) == "**dau**: —"


class TestTelemetry:
    def test_query_type_persisted(self, tmp_path):
        from memory.store import log_run, get_run
        path = str(tmp_path / "mem.db")
        run_id = log_run("what was dau?", path=path, query_type="lookup")
        row = get_run(run_id, path=path)
        assert row["query_type"] == "lookup"

    def test_migration_adds_column_to_existing_db(self, tmp_path):
        import sqlite3
        from memory.store import init_db
        path = str(tmp_path / "old.db")
        init_db(path)
        with sqlite3.connect(path) as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()}
        assert "query_type" in cols


class TestLookupHeuristicScope:
    """The regex override can qualify a run for the gate-free fast path, so a
    comparison or a cut must never read as a lookup, even when the sentence
    opens like one."""

    @pytest.mark.parametrize("task", [
        "what was the average revenue per user by variant",
        "what is the total revenue vs last month",
        "how many users per region",
        "what was the conversion for treatment",
        "show me total sales over time",
        "what is the number of orders split by platform",
    ])
    def test_comparisons_and_cuts_are_not_lookups(self, task):
        from agents.analyze.nodes_intent import _is_lookup_task
        assert not _is_lookup_task(task)

    @pytest.mark.parametrize("task", [
        "how many TVs were sold?",
        "what was the total revenue",
        "list the top 10 customers",
        "How many enterprise accounts in EMEA signed up yesterday?",
    ])
    def test_plain_retrievals_still_are(self, task):
        from agents.analyze.nodes_intent import _is_lookup_task
        assert _is_lookup_task(task)
