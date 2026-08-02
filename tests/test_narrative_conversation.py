"""
tests/test_narrative_conversation.py — the narrative request must never end on
an assistant turn.

`generate_narrative` appends its own output to `conversation_history`, and that
history is appended after the task prompt. Only the audit path appended a
following user turn, so an analyst-requested revision produced a request whose
last message was the previous narrative. The API reads that as a prefill:

  * 400 on Sonnet 4.6+ / Opus 4.6+ — which is why FAST_MODEL could not move;
  * on Haiku 4.5 it is *accepted*, and the model continues the old narrative
    instead of rewriting it. That one degrades every revision silently.

These tests cover `_conversation_turns` directly and then assert on the message
array `generate_narrative` actually builds, since the invariant belongs to the
request, not to the helper.
"""
from __future__ import annotations

import pytest

from agents.analyze.nodes_narrative import (
    _MAX_HISTORY_TURNS,
    _MAX_TURN_CHARS,
    _REVISION_REQUEST,
    _conversation_turns,
)


def _roles(turns):
    return [t["role"] for t in turns]


# ── The helper ────────────────────────────────────────────────────────────────

def test_empty_history_stays_empty():
    """A first pass has no history, and the task prompt is already a user turn."""
    assert _conversation_turns([]) == []


def test_trailing_assistant_gets_an_instruction_turn():
    """The regression: one narrative, analyst declines, next pass would prefill."""
    turns = _conversation_turns([{"role": "assistant", "content": "draft one"}])
    assert _roles(turns) == ["assistant", "user"]
    assert turns[-1]["content"] == _REVISION_REQUEST


def test_injected_turn_asks_for_a_replacement_not_a_continuation():
    """A filler turn would fix the 400 and leave the semantic bug in place."""
    assert "not a continuation" in _REVISION_REQUEST.lower()


def test_audit_corrected_history_is_left_alone():
    """The audit path already ends on a user turn; nothing should be added."""
    history = [
        {"role": "assistant", "content": "draft one"},
        {"role": "user", "content": "fix the confidence interval"},
    ]
    assert _conversation_turns(history) == history


def test_repeated_assistant_turns_collapse_to_the_latest():
    """Two narratives with no request between them: the later one supersedes."""
    turns = _conversation_turns([
        {"role": "assistant", "content": "draft one"},
        {"role": "assistant", "content": "draft two"},
    ])
    assert _roles(turns) == ["assistant", "user"]
    assert turns[0]["content"] == "draft two"


def test_repeated_user_turns_merge_rather_than_drop():
    """Both are instructions — dropping either loses a correction."""
    turns = _conversation_turns([
        {"role": "assistant", "content": "draft"},
        {"role": "user", "content": "fix the CI"},
        {"role": "user", "content": "and the lift number"},
    ])
    assert _roles(turns) == ["assistant", "user"]
    assert "fix the CI" in turns[-1]["content"]
    assert "and the lift number" in turns[-1]["content"]


def test_leading_user_turn_is_dropped():
    """History is appended after a user block, so it must open on assistant."""
    turns = _conversation_turns([
        {"role": "user", "content": "stray"},
        {"role": "assistant", "content": "draft"},
    ])
    assert _roles(turns) == ["assistant", "user"]
    assert turns[0]["content"] == "draft"


def test_malformed_turns_are_skipped():
    turns = _conversation_turns([
        {"role": "system", "content": "nope"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "draft"},
        {"content": "no role"},
    ])
    assert _roles(turns) == ["assistant", "user"]


def test_turn_content_is_capped():
    turns = _conversation_turns([{"role": "assistant", "content": "x" * 50_000}])
    assert len(turns[0]["content"]) == _MAX_TURN_CHARS


def test_history_is_trimmed_in_pairs():
    """Trimming one at a time would break alternation or the assistant opening."""
    history = []
    for i in range(10):
        history.append({"role": "assistant", "content": f"draft {i}"})
        history.append({"role": "user", "content": f"revise {i}"})
    turns = _conversation_turns(history)
    assert len(turns) <= _MAX_HISTORY_TURNS
    assert turns[0]["role"] == "assistant"
    assert _roles(turns) == ["assistant", "user"] * (len(turns) // 2)
    # The newest exchange survives; the oldest is what goes.
    assert turns[-1]["content"] == "revise 9"


@pytest.mark.parametrize("history", [
    [],
    [{"role": "assistant", "content": "a"}],
    [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}],
    [{"role": "user", "content": "a"}],
])
def test_output_is_always_alternating_and_user_terminated(history):
    turns = _conversation_turns(history)
    if turns:
        assert turns[0]["role"] == "assistant"
        assert turns[-1]["role"] == "user"
        assert all(a["role"] != b["role"] for a, b in zip(turns, turns[1:]))


def test_normalisation_is_idempotent():
    """The result is written back to state, so a second pass must not drift."""
    once = _conversation_turns([{"role": "assistant", "content": "draft"}])
    assert _conversation_turns(once) == once


# ── The request that actually goes out ────────────────────────────────────────

class _StubResponse:
    def __init__(self, text: str):
        # A thinking block first, as any adaptive-thinking model returns — the
        # node must read past it. See tests/test_llm_response.py.
        self.content = [
            type("Block", (), {"type": "thinking", "thinking": "..."})(),
            type("Block", (), {"type": "text", "text": text})(),
        ]
        self.usage = type("Usage", (), {
            "input_tokens": 1, "output_tokens": 1,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        })()


def _run_narrative(monkeypatch, history):
    """Call generate_narrative with a stubbed client; return the messages sent."""
    import agents.analyze.nodes_narrative as nn

    sent: list[list[dict]] = []

    class _Messages:
        def create(self, **kwargs):
            sent.append(kwargs["messages"])
            # Call 1 is the narrative; call 2 is the audit, which parses JSON.
            # An inert audit result keeps the correction path out of the way.
            if len(sent) == 1:
                return _StubResponse("polished narrative")
            return _StubResponse('{"findings": [], "corrected_narrative": ""}')

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(nn, "_anthropic_client", lambda: _Client())
    out = nn.generate_narrative({
        "analysis_mode": "general",
        "query_type": "lookup",
        "task": "what happened to signups",
        "conversation_history": history,
    })
    return sent[0], out


def test_request_never_ends_on_an_assistant_turn(monkeypatch):
    """End to end: one prior narrative, no audit correction — the 400 case."""
    messages, _ = _run_narrative(monkeypatch, [{"role": "assistant", "content": "draft one"}])
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == _REVISION_REQUEST
    assert all(a["role"] != b["role"] for a, b in zip(messages, messages[1:]))


def test_first_pass_sends_a_single_user_message(monkeypatch):
    messages, _ = _run_narrative(monkeypatch, [])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_stored_history_is_written_back_normalised(monkeypatch):
    """Otherwise the next pass re-derives from a malformed list and drifts."""
    _, out = _run_narrative(monkeypatch, [{"role": "assistant", "content": "draft one"}])
    history = out["conversation_history"]
    assert _roles(history) == ["assistant", "user", "assistant"]
    assert history[0]["content"] == "draft one"
    assert history[-1]["content"] == "polished narrative"


def test_revision_loop_does_not_accumulate_assistant_turns(monkeypatch):
    """Two declines in a row used to leave [assistant, assistant] in history."""
    history: list[dict] = []
    for _ in range(4):
        messages, out = _run_narrative(monkeypatch, history)
        assert messages[-1]["role"] == "user", _roles(messages)
        history = out["conversation_history"]
        assert all(a["role"] != b["role"] for a, b in zip(history, history[1:])), _roles(history)


def test_audit_call_budgets_for_a_thinking_block(monkeypatch):
    """The audit's max_tokens must cover thinking + JSON on adaptive-thinking
    models — 2048 starved the JSON once on claude-sonnet-5 (JSONDecodeError,
    silently skipped audit). Pin the configurable budget and that the call
    actually uses it."""
    import agents.analyze.node_shared as shared
    import agents.analyze.nodes_narrative as nn

    assert shared._MAX_TOKENS_AUDIT >= 8192

    calls: list[dict] = []

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _StubResponse("polished narrative")
            return _StubResponse('{"findings": [], "corrected_narrative": ""}')

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(nn, "_anthropic_client", lambda: _Client())
    nn.generate_narrative({
        "analysis_mode": "general",
        "query_type": "lookup",
        "task": "what happened to signups",
        "conversation_history": [],
    })
    audit_call = calls[1]
    assert audit_call["max_tokens"] == shared._MAX_TOKENS_AUDIT
