"""
tests/test_intent_concurrency.py — resolve_task_intent's merged config inference.

The retired infer_metric_config node ran AFTER intent and overwrote its
metric_config wholesale on every upload, discarding the task-informed metric
resolution. The merged node runs the two LLM calls concurrently (their inputs
are independent: task+schema vs schema alone) and applies intent's resolution
ON TOP of the schema-inferred base.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.analyze.nodes_intent as ni
from config.analysis_config import load_metric_config


SCHEMA = """TABLE: events  -- 1,000 rows
user_id VARCHAR
revenue_usd DOUBLE
sessions BIGINT
"""


def _intent_result(**overrides) -> dict:
    result = {
        "analysis_mode":       "general",
        "primary_metric":      "revenue_usd",
        "metric_direction":    "increase",
        "covariate":           "sessions",
        "guardrail_metrics":   [],
        "ambiguous":           False,
        "clarifying_question": None,
        "query_type":          "exploratory",
        "reasoning":           "test",
    }
    result.update(overrides)
    return result


def _upload_state(**overrides) -> dict:
    state = {
        "task":           "how much revenue did we make?",
        "schema_context": SCHEMA,
        "duckdb_path":    "/tmp/upload.db",
        "analysis_mode":  "general",
    }
    state.update(overrides)
    return state


class TestUploadInference:
    def test_intent_resolution_survives_on_uploads(self, monkeypatch):
        """The old ordering discarded intent's metric on every upload."""
        inferred = load_metric_config().model_copy(
            update={"primary_metric": "sessions", "events_table": "events"}
        )
        monkeypatch.setattr(ni, "_llm_resolve_intent", lambda *a: _intent_result())
        monkeypatch.setattr(ni, "_llm_infer_config", lambda *a: inferred)

        out = ni.resolve_task_intent(_upload_state())
        # Intent's task-informed metric wins over the schema-only guess...
        assert out["metric_config"].primary_metric == "revenue_usd"
        # ...but the inferred base's table mapping survives underneath.
        assert out["metric_config"].events_table == "events"

    def test_calls_run_concurrently(self, monkeypatch):
        """Both LLM calls must be in flight at once — that is the entire point."""
        barrier = threading.Barrier(2, timeout=5)

        def _intent(*a):
            barrier.wait()  # deadlocks (and times out) unless infer runs in parallel
            return _intent_result()

        def _infer(*a):
            barrier.wait()
            return load_metric_config()

        monkeypatch.setattr(ni, "_llm_resolve_intent", _intent)
        monkeypatch.setattr(ni, "_llm_infer_config", _infer)

        out = ni.resolve_task_intent(_upload_state())
        assert out["metric_config"].primary_metric == "revenue_usd"

    def test_no_inference_without_upload(self, monkeypatch):
        monkeypatch.setattr(ni, "_llm_resolve_intent", lambda *a: _intent_result())
        def _boom(*a):
            raise AssertionError("schema inference ran for a non-upload")
        monkeypatch.setattr(ni, "_llm_infer_config", _boom)

        out = ni.resolve_task_intent(_upload_state(duckdb_path=None))
        assert out["metric_config"].primary_metric == "revenue_usd"

    def test_no_inference_with_certified_pack(self, monkeypatch):
        monkeypatch.setattr(ni, "_llm_resolve_intent", lambda *a: _intent_result())
        def _boom(*a):
            raise AssertionError("schema inference ran despite a metric pack")
        monkeypatch.setattr(ni, "_llm_infer_config", _boom)

        out = ni.resolve_task_intent(
            _upload_state(metric_pack_id="pack-1", metric_config=load_metric_config())
        )
        assert out["metric_config"] is not None


class TestInferConfigHelper:
    def test_empty_schema_returns_defaults(self):
        assert ni._llm_infer_config("") == load_metric_config()

    def test_llm_failure_returns_defaults(self, monkeypatch):
        def _boom():
            raise RuntimeError("api down")
        monkeypatch.setattr(ni, "_anthropic_client", _boom)
        assert ni._llm_infer_config(SCHEMA) == load_metric_config()


class TestGraphShape:
    def test_infer_node_is_gone(self):
        from agents.analyze.graph import build_graph
        g = build_graph()
        assert "infer_metric_config" not in g.get_graph().nodes
        assert "resolve_task_intent" in g.get_graph().nodes
