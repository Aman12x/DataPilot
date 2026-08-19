"""
tests/test_result_too_large_routing.py — an over-ceiling general-mode query
stops at query_gate with the row count, instead of degrading to an empty
frame and a "no data" narrative.

6490e10 made db.query refuse results above _MAX_MATERIALIZE_ROWS. In ab_test
mode the canonical fallback raises, so the refusal is loud. In general mode
the retry loop's terminal path returned an empty DataFrame with only a log
line — the refusal message never reached the analyst.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.analyze.nodes_sql as ns  # noqa: E402
from agents.analyze.graph import _route_after_execute_query  # noqa: E402
from tools import db_tools  # noqa: E402
from tools.db_tools import DBConnection  # noqa: E402


def _state(tmp_duckdb: str) -> dict:
    return {
        "generated_sql":  "SELECT user_id, dau_flag FROM events",
        "analysis_mode":  "general",
        "db_backend":     "duckdb",
        "duckdb_path":    tmp_duckdb,
        "schema_context": "TABLE: events\nuser_id VARCHAR\ndau_flag INTEGER\n",
        "task":           "list everything",
    }


def test_oversized_general_query_blocks_at_query_gate(tmp_duckdb, monkeypatch):
    monkeypatch.setattr(db_tools, "_MAX_MATERIALIZE_ROWS", 1)
    monkeypatch.setattr(ns, "_db_conn", lambda state: DBConnection("duckdb", path=tmp_duckdb))
    # The correction LLM gives up (unchanged SQL) — the terminal path.
    monkeypatch.setattr(ns, "_llm_correct_sql", lambda sql, *a: sql)

    out = ns.execute_query(_state(tmp_duckdb))

    assert isinstance(out["query_result"], pd.DataFrame) and out["query_result"].empty
    warnings = out["sql_validation_warnings"]
    assert len(warnings) == 1
    assert "above the 1-row limit" in warnings[0]
    assert "Aggregate in SQL" in warnings[0]

    routed = _route_after_execute_query({**_state(tmp_duckdb), **out})
    assert routed == "query_gate"


def test_other_general_failures_keep_the_old_empty_frame_path(tmp_duckdb, monkeypatch):
    """Scope guard: only the materialisation refusal is promoted to a block."""
    monkeypatch.setattr(ns, "_db_conn", lambda state: DBConnection("duckdb", path=tmp_duckdb))
    monkeypatch.setattr(ns, "_llm_correct_sql", lambda sql, *a: sql)
    st = {**_state(tmp_duckdb), "generated_sql": "SELECT nope FROM events"}

    out = ns.execute_query(st)
    assert out["query_result"].empty
    assert not out.get("sql_validation_warnings")
    assert _route_after_execute_query({**st, **out}) == "describe_data"
