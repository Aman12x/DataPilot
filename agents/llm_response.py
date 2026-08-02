"""
agents/llm_response.py — read text out of a Messages API response safely.

Eight call sites did `response.content[0].text`. That holds only while the first
content block is a text block, which stopped being true the moment FAST_MODEL
was pointed at a model with adaptive thinking on by default: `content[0]` is a
`ThinkingBlock`, and the whole node dies with

    AttributeError: 'ThinkingBlock' object has no attribute 'text'

The failure is an AttributeError deep inside a node, not an API error, so it
reads like a bug in the node rather than a model-compatibility problem.

Concatenating every text block is right regardless of model: a response may
legitimately carry several, and non-text blocks (thinking, tool_use) are never
the answer.
"""
from __future__ import annotations

from typing import Any


def response_text(response: Any) -> str:
    """Concatenate the text blocks of a Messages API response.

    Returns "" when the response carries no text at all — a `max_tokens` stop
    during a thinking block can produce exactly that, and callers already handle
    empty output better than they handle an AttributeError.
    """
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        # Match on the block type rather than duck-typing `.text`: it is the
        # field the API documents, and it keeps a future block type that happens
        # to carry text from being spliced into the answer.
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
