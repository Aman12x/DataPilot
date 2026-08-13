"""
tests/test_prompt_cache_alignment.py — prompt-cache prefix discipline.

Prompt caching only hits when the prefix is byte-identical across calls.
Before this suite existed, intent cached the full schema, generate_sql cached
a table-filtered variant, the narrative cached a truncated one, and the
correction/inference calls sent flat prompts with no cache_control at all —
five different byte-sequences, so every cache write was an orphan and the
runs table showed zero cache reads run after run.

These tests pin the invariant: every LLM call that ships the schema ships
_cached_schema_block's bytes, marked ephemeral, in the same position.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.analyze import node_shared
from agents.analyze.node_shared import (
    _build_cached_messages,
    _cached_schema_block,
)


SCHEMA = "TABLE: events  -- 1,000 rows\nuser_id VARCHAR\nrevenue DOUBLE\n"


class _FakeResponse:
    class _Usage:
        input_tokens = 10
        output_tokens = 5
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
    usage = _Usage()
    stop_reason = "end_turn"
    content = []


def _capture_messages(monkeypatch, module):
    """Stub the Anthropic client in `module`; record every messages.create call."""
    calls: list[dict] = []

    class _Messages:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("stop after capture")  # callers all fail safe

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(module, "_anthropic_client", lambda: _Client())
    return calls


def _schema_blocks(call: dict) -> list[str]:
    """Texts of cache_control-marked blocks containing the schema wrapper."""
    blocks = call["messages"][0]["content"]
    return [
        b["text"] for b in blocks
        if isinstance(b, dict)
        and b.get("cache_control")
        and "USER_DATABASE_SCHEMA" in b.get("text", "")
    ]


class TestCanonicalBlock:
    def test_deterministic(self):
        assert _cached_schema_block(SCHEMA) == _cached_schema_block(SCHEMA)

    def test_caps_at_shared_constant(self):
        big = "x" * (node_shared._SCHEMA_PROMPT_MAX_CHARS + 5000)
        a = _cached_schema_block(big)
        b = _cached_schema_block(big + "different tail beyond the cap")
        assert a == b  # identical after the cap → still one cache entry

    def test_empty_schema_produces_no_block(self):
        assert _cached_schema_block("") == ""

    def test_build_marks_schema_ephemeral(self):
        msgs = _build_cached_messages(_cached_schema_block(SCHEMA), "", "task")
        schema_blocks = [
            b for b in msgs[0]["content"]
            if b.get("cache_control") == {"type": "ephemeral"}
            and "USER_DATABASE_SCHEMA" in b["text"]
        ]
        assert len(schema_blocks) == 1


class TestEveryCallSendsTheSameBytes:
    """The cross-call property the cache depends on."""

    def test_intent_and_correction_share_the_prefix(self, monkeypatch):
        import agents.analyze.nodes_intent as ni

        intent_calls = _capture_messages(monkeypatch, ni)
        ni._llm_resolve_intent("task?", SCHEMA, __import__("config.analysis_config", fromlist=["load_metric_config"]).load_metric_config())

        shared_calls = _capture_messages(monkeypatch, node_shared)
        node_shared._llm_correct_sql("SELECT 1", "boom", SCHEMA, "task?")

        infer_calls = _capture_messages(monkeypatch, ni)
        ni._llm_infer_config(SCHEMA)

        blocks = [
            _schema_blocks(intent_calls[0]),
            _schema_blocks(shared_calls[0]),
            _schema_blocks(infer_calls[0]),
        ]
        for b in blocks:
            assert len(b) == 1, "call is missing its cached schema block"
        assert blocks[0] == blocks[1] == blocks[2], (
            "schema blocks differ across calls — the cache can never hit"
        )
        assert blocks[0][0] == _cached_schema_block(SCHEMA)

    def test_correction_prompt_no_longer_inlines_schema(self):
        from agents.analyze.prompts import SQL_CORRECTION_PROMPT
        assert "{schema_context}" not in SQL_CORRECTION_PROMPT

    def test_inference_prompt_no_longer_inlines_schema(self):
        from agents.analyze.prompts import SCHEMA_CONFIG_INFERENCE_PROMPT
        assert "{schema_context}" not in SCHEMA_CONFIG_INFERENCE_PROMPT
