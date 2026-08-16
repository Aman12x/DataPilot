"""
A retired MODEL pin must not silently disable intent resolution.

_llm_resolve_intent falls back to safe_default on any exception, which is
right for a malformed reply but wrong for anthropic.NotFoundError: that is a
permanent config error, and swallowing it meant every run for weeks got the
"Defaulting to current metric config" path with only a warning. The resolver
now retries once on FAST_MODEL and logs at ERROR.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import anthropic
import httpx
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.analyze.nodes_intent as ni
from config.analysis_config import load_metric_config

SCHEMA = "TABLE: events  -- 10 rows\nuser_id VARCHAR\nrevenue_usd DOUBLE\n"

_REPLY = {
    "analysis_mode": "ab_test",
    "primary_metric": "revenue_usd",
    "metric_direction": "increase",
    "covariate": None,
    "guardrail_metrics": [],
    "ambiguous": False,
    "clarifying_question": None,
    "reasoning": "revenue is the only metric column",
}


def _not_found() -> anthropic.NotFoundError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(404, request=req, json={"error": {"message": "model: retired-model"}})
    return anthropic.NotFoundError("model not found", response=resp, body=None)


@contextmanager
def _fake_trace(*_a, **_k):
    yield SimpleNamespace(update=lambda _resp: {"usd": 0.0})


@pytest.fixture
def client(monkeypatch):
    calls: list[str] = []

    class _Messages:
        def create(self, *, model, **_kw):
            calls.append(model)
            if model == "retired-model":
                raise _not_found()
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(_REPLY))])

    fake = SimpleNamespace(messages=_Messages())
    monkeypatch.setattr(ni, "_anthropic_client", lambda: fake)
    monkeypatch.setattr(ni, "trace_generation", _fake_trace)
    monkeypatch.setattr(ni, "_model", lambda: "retired-model")
    monkeypatch.setattr(ni, "_fast_model", lambda: "workhorse-model")
    return calls


def test_stale_model_pin_retries_on_fast_model(client, caplog):
    mc = load_metric_config()
    with caplog.at_level("ERROR"):
        result, _ = ni._llm_resolve_intent("which variant made more revenue?", SCHEMA, mc)

    assert client == ["retired-model", "workhorse-model"]
    assert result["primary_metric"] == "revenue_usd"
    assert result["reasoning"] != "Defaulting to current metric config."
    assert any("stale MODEL env pin" in r.getMessage() for r in caplog.records)


def test_healthy_pin_makes_one_call(client, monkeypatch):
    monkeypatch.setattr(ni, "_model", lambda: "workhorse-model")
    result, _ = ni._llm_resolve_intent("which variant made more revenue?", SCHEMA, load_metric_config())
    assert client == ["workhorse-model"]
    assert result["primary_metric"] == "revenue_usd"
