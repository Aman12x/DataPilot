"""Table selection before SQL generation (future-work item 7)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.analyze.node_shared as ns
from config.analysis_config import load_metric_config

WIDE_SCHEMA = "\n\n".join(
    [f"TABLE: table_{i}  -- {i * 100:,} rows\n  id VARCHAR\n  value DOUBLE" for i in range(9)]
    + ["TABLE: events  -- 284,797 rows\n  user_id VARCHAR  -- unique user\n  dau_flag INTEGER",
       "TABLE: experiment  -- 20,000 rows\n  user_id VARCHAR\n  variant VARCHAR"]
)

NARROW_SCHEMA = (
    "TABLE: events  -- 284,797 rows\n  user_id VARCHAR\n  dau_flag INTEGER\n\n"
    "TABLE: experiment  -- 20,000 rows\n  user_id VARCHAR\n  variant VARCHAR"
)


class _FakeClient:
    def __init__(self, reply: str, calls: list):
        self._reply = reply
        self._calls = calls

    @property
    def messages(self):
        outer = self

        class M:
            def create(self, **kw):
                outer._calls.append(kw)
                block = SimpleNamespace(type="text", text=outer._reply)
                return SimpleNamespace(
                    content=[block], stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                          cache_read_input_tokens=0,
                                          cache_creation_input_tokens=0),
                )
        return M()


def _patch_client(monkeypatch, reply: str):
    calls: list = []
    monkeypatch.setattr(ns, "_anthropic_client", lambda: _FakeClient(reply, calls))
    return calls


def test_block_parsing_and_summaries():
    blocks = ns._schema_table_blocks(NARROW_SCHEMA)
    assert set(blocks) == {"events", "experiment"}
    summary = ns._table_summaries(blocks)
    assert "- events (284,797 rows): user_id, dau_flag" in summary


def test_preamble_is_preserved_in_filtered_context():
    ctx = "DIALECT: duckdb\n\n" + NARROW_SCHEMA
    filtered = ns._filter_schema_context(ctx, {"events"})
    assert filtered.startswith("DIALECT: duckdb")
    assert "TABLE: events" in filtered and "TABLE: experiment" not in filtered


def test_narrow_schema_skips_the_llm_call(monkeypatch):
    calls = _patch_client(monkeypatch, '["events"]')
    ctx, tables = ns._select_relevant_tables(
        "any task", NARROW_SCHEMA, load_metric_config(), "general")
    assert calls == []              # no LLM call for 2 tables
    assert ctx == NARROW_SCHEMA
    assert set(tables) == {"events", "experiment"}


def test_wide_schema_prunes_to_selection(monkeypatch):
    _patch_client(monkeypatch, '["events", "experiment"]')
    ctx, tables = ns._select_relevant_tables(
        "dau by variant", WIDE_SCHEMA, load_metric_config(), "general")
    assert set(tables) == {"events", "experiment"}
    assert "TABLE: events" in ctx and "TABLE: table_3" not in ctx


def test_ab_mode_forces_canonical_tables_back_in(monkeypatch):
    # The selector "forgot" the experiment table; A/B mode must restore it.
    _patch_client(monkeypatch, '["events"]')
    ctx, tables = ns._select_relevant_tables(
        "dau by variant", WIDE_SCHEMA, load_metric_config(), "ab_test")
    assert "experiment" in tables
    assert "TABLE: experiment" in ctx


def test_selection_of_unknown_tables_falls_back_to_full(monkeypatch):
    _patch_client(monkeypatch, '["not_a_real_table"]')
    ctx, _ = ns._select_relevant_tables(
        "task words", WIDE_SCHEMA, load_metric_config(), "general")
    assert ctx == WIDE_SCHEMA


def test_malformed_reply_falls_back_to_full(monkeypatch):
    _patch_client(monkeypatch, "I think events is the one you want")
    ctx, _ = ns._select_relevant_tables(
        "task words", WIDE_SCHEMA, load_metric_config(), "general")
    assert ctx == WIDE_SCHEMA


def test_truncated_reply_falls_back_to_full(monkeypatch):
    calls: list = []

    class Trunc(_FakeClient):
        @property
        def messages(self):
            outer = self

            class M:
                def create(self, **kw):
                    block = SimpleNamespace(type="text", text='["events"')
                    return SimpleNamespace(
                        content=[block], stop_reason="max_tokens",
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                              cache_read_input_tokens=0,
                                              cache_creation_input_tokens=0),
                    )
            return M()

    import agents.analyze.node_shared as mod
    mod_client = Trunc("", calls)
    mod_orig = mod._anthropic_client
    mod._anthropic_client = lambda: mod_client
    try:
        ctx, _ = ns._select_relevant_tables(
            "task words", WIDE_SCHEMA, load_metric_config(), "general")
    finally:
        mod._anthropic_client = mod_orig
    assert ctx == WIDE_SCHEMA


def test_code_fenced_reply_is_accepted(monkeypatch):
    _patch_client(monkeypatch, '```json\n["events"]\n```')
    ctx, tables = ns._select_relevant_tables(
        "task words", WIDE_SCHEMA, load_metric_config(), "general")
    assert tables == ["events"]
    assert "TABLE: table_1" not in ctx


def test_generate_sql_prompt_excludes_pruned_tables(monkeypatch, tmp_path):
    """Seam test: on a wide schema, decoy tables never reach the SQL prompt."""
    import agents.analyze.nodes_sql as nsql

    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "m.db"))
    captured: list = []

    class Seq:
        """First call answers table selection, second the SQL generation."""
        @property
        def messages(self):
            outer = self

            class M:
                def create(self, **kw):
                    captured.append(kw)
                    text = '["events", "experiment"]' if len(captured) == 1 else "```sql\nSELECT 1\n```"
                    block = SimpleNamespace(type="text", text=text)
                    return SimpleNamespace(
                        content=[block], stop_reason="end_turn",
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                              cache_read_input_tokens=0,
                                              cache_creation_input_tokens=0),
                    )
            return M()

    monkeypatch.setattr(ns, "_anthropic_client", lambda: Seq())
    monkeypatch.setattr(nsql, "_anthropic_client", lambda: Seq())

    out = nsql.generate_sql({
        "task": "How many users are in each variant?",
        "analysis_mode": "general",
        "db_backend": "duckdb",
        "schema_context": WIDE_SCHEMA,
        "relevant_history": [],
        "user_id": "u1",
    })
    assert out["generated_sql"] == "SELECT 1"
    blocks = captured[-1]["messages"][0]["content"]
    # Task prompt (final, uncached block): only the selected tables — this is
    # the section the model is told to work from.
    task_block = blocks[-1]["text"]
    assert "TABLE: events" in task_block
    assert "table_5" not in task_block
    assert "reference only" in task_block  # pruning directive present
    # Cached prefix: the CANONICAL full schema — deliberately including pruned
    # tables, because the cache only hits when every call sends the same bytes.
    cached_schema = [
        b["text"] for b in blocks
        if b.get("cache_control") and "USER_DATABASE_SCHEMA" in b.get("text", "")
    ]
    assert len(cached_schema) == 1
    assert "table_5" in cached_schema[0]
