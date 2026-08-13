"""
tests/test_narrative_streaming.py — the narrative draft streams to watchers.

generate_narrative's draft call is the longest LLM call in a run (~60-80s
measured); it now streams, emitting batched text deltas through stream_hub so
the SSE stream can show the report being written. Invariants defended here:

1. Deltas reassemble to exactly the draft text, prefixed by narrative_start
   (which lets a revision replace a stale draft instead of appending).
2. Nothing is emitted — and nothing crashes — when no emitter is registered.
3. A failing emitter never breaks generation.
4. The metered client wrapper prices streamed calls exactly once (the
   "every LLM call is metered" invariant would otherwise silently lose
   every narrative to the budget).
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.analyze import stream_hub


@pytest.fixture(autouse=True)
def _clean_hub():
    yield
    stream_hub._emitters.clear()


class _StubResponse:
    def __init__(self, text: str):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = type("U", (), {
            "input_tokens": 1, "output_tokens": 1,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        })()


class _FakeStream:
    """Yields the draft in small chunks, like the real SDK stream."""

    def __init__(self, response, chunk: int = 7):
        self._response = response
        self._chunk = chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        text = self._response.content[0].text
        for i in range(0, len(text), self._chunk):
            yield text[i:i + self._chunk]

    def get_final_message(self):
        return self._response


def _stub_client(draft_text: str):
    calls: list[dict] = []

    class _Messages:
        def stream(self, **kwargs):
            calls.append(kwargs)
            return _FakeStream(_StubResponse(draft_text))

        def create(self, **kwargs):
            calls.append(kwargs)
            return _StubResponse('{"passed": true, "findings": [], "corrected_narrative": ""}')

    class _Client:
        messages = _Messages()

    return _Client(), calls


DRAFT = "The treatment reduced DAU by 1.2 points. " * 20  # > one flush batch


def _run(monkeypatch, run_id: str):
    import agents.analyze.nodes_narrative as nn
    client, calls = _stub_client(DRAFT)
    monkeypatch.setattr(nn, "_anthropic_client", lambda: client)
    out = nn.generate_narrative({
        "analysis_mode": "general",
        "task": "what happened to DAU",
        "run_id": run_id,
        "conversation_history": [],
    })
    return out, calls


class TestDeltaEmission:
    def test_deltas_reassemble_to_the_draft(self, monkeypatch):
        events: list[dict] = []
        stream_hub.register("run-1", events.append)

        out, _ = _run(monkeypatch, "run-1")

        assert events[0] == {"type": "narrative_start"}
        deltas = [e for e in events[1:] if e["type"] == "narrative_delta"]
        assert deltas, "no deltas emitted"
        assert "".join(d["text"] for d in deltas) == DRAFT
        # Batched: far fewer events than SDK chunks (len(DRAFT)/7 chunks)
        assert len(deltas) < len(DRAFT) / 7 / 2
        assert out["narrative_draft"] == DRAFT.strip()

    def test_no_emitter_is_a_silent_no_op(self, monkeypatch):
        out, _ = _run(monkeypatch, "run-unwatched")
        assert out["narrative_draft"] == DRAFT.strip()

    def test_failing_emitter_never_breaks_generation(self, monkeypatch):
        def _boom(payload):
            raise RuntimeError("reader went away")
        stream_hub.register("run-2", _boom)

        out, _ = _run(monkeypatch, "run-2")
        assert out["narrative_draft"] == DRAFT.strip()

    def test_unregister_stops_delivery(self):
        events: list[dict] = []
        stream_hub.register("run-3", events.append)
        stream_hub.unregister("run-3")
        stream_hub.emit("run-3", {"type": "narrative_delta", "text": "x"})
        assert events == []


class TestMeteredStreaming:
    def test_streamed_call_is_priced_exactly_once(self, monkeypatch):
        from agents.analyze import node_shared
        from agents import spend

        recorded: list[tuple] = []
        monkeypatch.setattr(spend, "record", lambda model, resp: recorded.append((model, resp)))

        response = _StubResponse("hello")

        class _RawMessages:
            def stream(self, **kwargs):
                return _FakeStream(response)

        metered = node_shared._MeteredMessages(_RawMessages())
        with metered.stream(model="claude-sonnet-5", max_tokens=10, messages=[]) as s:
            list(s.text_stream)
            final = s.get_final_message()
            s.get_final_message()  # a second read must not double-charge

        assert final is response
        assert len(recorded) == 1
        assert recorded[0][0] == "claude-sonnet-5"
