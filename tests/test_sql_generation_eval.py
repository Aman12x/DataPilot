"""SQL-generation golden-question eval — pytest wrapper.

The harness itself lives in evals/sql_generation_eval.py and calls the LLM
live, so the scored run is marked slow (nightly/manual, not per-PR). The
fixture-integrity tests below are fast and always run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import sql_generation_eval as sge  # noqa: E402


def test_fixture_tables_exist_in_demo_db():
    """Every expected table in a demo-DB question must exist, or the tables
    stage can never pass and the eval silently measures nothing."""
    import duckdb

    con = duckdb.connect(sge.DEMO_DB, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    for q in sge.QUESTIONS:
        if not q.wide:
            missing = set(q.expected_tables) - tables
            assert not missing, f"{q.qid}: expected tables missing from demo DB: {missing}"


def test_wide_db_builder_creates_decoys(tmp_path):
    import duckdb

    dest = str(tmp_path / "wide.db")
    sge.build_wide_db(dest)
    con = duckdb.connect(dest, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert set(sge.DECOY_TABLES) <= tables
    assert {"events", "experiment", "funnel", "metrics_daily"} <= tables


def test_predicates_are_total():
    """Predicates must return a bool for an empty frame rather than raise —
    a raising predicate would score as failure with a confusing detail."""
    empty = pd.DataFrame()
    for q in sge.QUESTIONS:
        result = None
        try:
            result = q.predicate(empty)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{q.qid} predicate raised on empty frame: {exc}")
        assert result in (True, False), f"{q.qid} predicate returned {result!r}"


@pytest.mark.slow
def test_sql_generation_meets_threshold():
    """Live run: strict pass rate must hold the baseline (LLM required)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(Path(sge.__file__)), "--threshold", "0.75"],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"strict pass rate under threshold:\n{proc.stdout[-2000:]}"
