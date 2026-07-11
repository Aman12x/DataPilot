"""
agents/analyze/prompt_safety.py — Delimiter wrapping for untrusted user content.

User tasks, uploaded schema metadata, and analyst free-text can contain
prompt-injection attempts. Wrap them in explicit delimiters before LLM calls.
"""
from __future__ import annotations

_USER_END = "<<<END_USER_CONTENT>>>"


def wrap_untrusted_content(text: str, *, label: str = "user") -> str:
    """Wrap analyst-supplied text so the model treats it as data, not instructions."""
    if not text:
        return f"<<<USER_{label.upper()}>>>\n<<<{_USER_END}>>>"
    sanitized = text.replace(_USER_END, "")
    return f"<<<USER_{label.upper()}>>>\n{sanitized}\n<<<{_USER_END}>>>"
