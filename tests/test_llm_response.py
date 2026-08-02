"""
tests/test_llm_response.py — `content[0]` is not always a text block.

Eight call sites read `response.content[0].text`. A model with adaptive thinking
enabled returns `[ThinkingBlock, TextBlock]`, so every one of them raised

    AttributeError: 'ThinkingBlock' object has no attribute 'text'

which surfaces as a bug in the node rather than a model-compatibility problem.
Verified live against claude-sonnet-5 before the fix.
"""
from __future__ import annotations

import types

import pytest

from agents.llm_response import response_text


def _block(kind: str, **fields):
    return types.SimpleNamespace(type=kind, **fields)


def _response(*blocks):
    return types.SimpleNamespace(content=list(blocks))


def test_plain_text_response():
    assert response_text(_response(_block("text", text="hello"))) == "hello"


def test_thinking_block_first_is_skipped():
    """The regression, as observed from claude-sonnet-5."""
    resp = _response(
        _block("thinking", thinking="deliberating"),
        _block("text", text='{"passed": true}'),
    )
    assert response_text(resp) == '{"passed": true}'


def test_multiple_text_blocks_are_concatenated():
    resp = _response(_block("text", text="one "), _block("text", text="two"))
    assert response_text(resp) == "one two"


def test_tool_use_blocks_are_skipped():
    resp = _response(
        _block("tool_use", name="run_sql", input={}),
        _block("text", text="done"),
    )
    assert response_text(resp) == "done"


def test_thinking_only_response_returns_empty_string():
    """A max_tokens stop inside thinking yields no text at all.

    Callers handle "" — they retry, or fall back to a template — far better
    than they handle an AttributeError.
    """
    assert response_text(_response(_block("thinking", thinking="..."))) == ""


@pytest.mark.parametrize("resp", [
    types.SimpleNamespace(content=[]),
    types.SimpleNamespace(content=None),
    types.SimpleNamespace(),
])
def test_missing_or_empty_content_is_not_fatal(resp):
    assert response_text(resp) == ""


def test_block_without_a_text_field_is_skipped():
    """Type says text but the field is absent — never raise from a log path."""
    assert response_text(_response(_block("text"))) == ""


def test_no_call_site_still_indexes_content_directly():
    """One missed site is a crash on the first response that thinks."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for sub in ("agents", "tools", "backend"):
        for path in (root / sub).rglob("*.py"):
            if path.name == "llm_response.py":
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if ".content[0]" in line:
                    offenders.append(f"{path.relative_to(root)}:{i}")
    assert not offenders, (
        "read the response with agents.llm_response.response_text instead:\n  "
        + "\n  ".join(offenders)
    )
