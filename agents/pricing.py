"""
agents/pricing.py — Anthropic list prices, USD per million tokens.

Kept separate from tracer.py so the budget enforcement in backend/api/budget.py
can price a call without pulling in Langfuse.

Prices are (input, output) per million tokens. Cache reads bill at 0.1x the
input rate and 5-minute cache writes at 1.25x, so both are derived rather than
listed per model.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

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
    # Opus 4.1 predates the $5/$25 Opus tier and is materially pricier.
    "claude-opus-4-1":   (15.00, 75.00),
    "claude-sonnet-5":   (3.00,  15.00),
    "claude-sonnet-4-6": (3.00,  15.00),
    "claude-sonnet-4-5": (3.00,  15.00),
    # Matches the dated snapshot claude-sonnet-4-20250514 by prefix.
    "claude-sonnet-4":   (3.00,  15.00),
    "claude-haiku-4-5":  (1.00,   5.00),
}

# An unrecognised model bills at the most expensive tier we know of rather than
# the cheapest: a budget that under-charges is worse than one that over-charges.
# Anchored to Opus 4.1's $15/$75 — the highest listed rate — not to the newest
# model's, which would silently under-bill an older, pricier one.
_FALLBACK = (15.00, 75.00)

_CACHE_READ_MULTIPLIER  = 0.10
_CACHE_WRITE_MULTIPLIER = 1.25


_warned_unknown: set[str] = set()


def rates(model: str) -> tuple[float, float, float, float]:
    """(input, output, cache_read, cache_write) $/Mtok for a model ID."""
    best = ""
    for prefix in _PRICES:
        if model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    if not best:
        # Warn once per model. Silent fallback is how a mispriced model hides:
        # spend still gets counted, but against a rate nobody chose.
        if model not in _warned_unknown:
            _warned_unknown.add(model)
            logger.warning(
                "No pricing entry for model %r — billing at the fallback rate "
                "($%.2f/$%.2f per Mtok). Add it to agents/pricing.py.",
                model, *_FALLBACK,
            )
        return (*_FALLBACK, _FALLBACK[0] * _CACHE_READ_MULTIPLIER,
                _FALLBACK[0] * _CACHE_WRITE_MULTIPLIER)
    inp, out = _PRICES[best]
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
