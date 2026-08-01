"""
agents/pricing.py — Anthropic list prices, USD per million tokens.

Kept separate from tracer.py so the budget enforcement in backend/api/budget.py
can price a call without pulling in Langfuse.

Prices are (input, output) per million tokens. Cache reads bill at 0.1x the
input rate and 5-minute cache writes at 1.25x, so both are derived rather than
listed per model.
"""
from __future__ import annotations

# Model ID prefix → (input $/Mtok, output $/Mtok). Matched by longest prefix so
# dated snapshots (claude-haiku-4-5-20251001) resolve to their family.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5":    (10.00, 50.00),
    "claude-mythos-5":   (10.00, 50.00),
    "claude-opus-5":     (5.00,  25.00),
    "claude-opus-4-8":   (5.00,  25.00),
    "claude-opus-4-7":   (5.00,  25.00),
    "claude-opus-4-6":   (5.00,  25.00),
    "claude-opus-4-5":   (5.00,  25.00),
    "claude-sonnet-5":   (3.00,  15.00),
    "claude-sonnet-4-6": (3.00,  15.00),
    "claude-sonnet-4-5": (3.00,  15.00),
    "claude-haiku-4-5":  (1.00,   5.00),
}

# An unrecognised model bills at the most expensive tier rather than the
# cheapest: a budget that under-charges is worse than one that over-charges.
_FALLBACK = (10.00, 50.00)

_CACHE_READ_MULTIPLIER  = 0.10
_CACHE_WRITE_MULTIPLIER = 1.25


def rates(model: str) -> tuple[float, float, float, float]:
    """(input, output, cache_read, cache_write) $/Mtok for a model ID."""
    best = ""
    for prefix in _PRICES:
        if model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    inp, out = _PRICES[best] if best else _FALLBACK
    return inp, out, inp * _CACHE_READ_MULTIPLIER, inp * _CACHE_WRITE_MULTIPLIER


def is_known_model(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _PRICES)


def cost_usd(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Price one API call.

    `input_tokens` from the Anthropic usage object is already the uncached
    remainder — cache reads and writes are reported separately and must not be
    subtracted from it again.
    """
    inp_rate, out_rate, read_rate, write_rate = rates(model)
    return (
        input_tokens       * inp_rate   / 1_000_000
        + output_tokens    * out_rate   / 1_000_000
        + cache_read_tokens  * read_rate  / 1_000_000
        + cache_write_tokens * write_rate / 1_000_000
    )


def cost_from_usage(model: str, usage: object) -> float:
    """Price a call from an anthropic.types.Usage (or any object with the fields)."""
    return cost_usd(
        model,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
