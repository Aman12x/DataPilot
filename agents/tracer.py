"""
agents/tracer.py — Langfuse observability setup for DataPilot.

Provides:
  - observe      : re-exported decorator for node-level tracing
  - trace_generation : context manager to log an Anthropic API call as a
                       Langfuse generation span (model, tokens, cost, I/O)
  - flush        : call at end of each run to ensure spans are flushed

Graceful no-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set —
tests and local dev without Langfuse credentials work without any changes.

Usage in nodes.py:
    from agents.tracer import observe, trace_generation

    @observe(name="run_cuped")
    def run_cuped_node(state: AgentState) -> AgentState:
        ...

    with trace_generation("generate_sql", model, prompt, max_tokens) as gen:
        response = client.messages.create(...)
        gen.update(response)   # logs token counts + cost
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── Initialise Langfuse (disabled automatically if keys are absent) ────────────

try:
    from langfuse import observe as _lf_observe, get_client as _lf_get_client
    _client = _lf_get_client()
    _LANGFUSE_ENABLED = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
except Exception:           # noqa: BLE001
    _LANGFUSE_ENABLED = False
    _client            = None
    _lf_observe        = None


# ── observe decorator ─────────────────────────────────────────────────────────

def observe(func=None, *, name: str | None = None, as_type: str = "span"):
    """
    Decorator that wraps a node function with a Langfuse span.

    Falls back to a no-op wrapper when Langfuse is disabled so that the
    decorated function still works without credentials.

    Usage:
        @observe(name="run_cuped")
        def run_cuped_node(state): ...
    """
    def decorator(fn):
        if _LANGFUSE_ENABLED and _lf_observe is not None:
            return _lf_observe(fn, name=name or fn.__name__, as_type=as_type)
        return fn  # no-op passthrough

    if func is not None:
        # Called as @observe without parentheses
        return decorator(func)
    return decorator


# ── GenerationContext: wraps a single Anthropic API call ─────────────────────

class GenerationContext:
    """
    Helper returned by trace_generation(). Call .update(response) after the
    Anthropic API call to log token counts and estimated cost.
    """

    # Sonnet pricing (USD per million tokens) — update if model changes
    _INPUT_COST_PER_M  = 3.00
    _OUTPUT_COST_PER_M = 15.00
    _CACHE_READ_PER_M  = 0.30
    _CACHE_WRITE_PER_M = 3.75

    def __init__(self, name: str, model: str, prompt: str, max_tokens: int):
        self.name       = name
        self.model      = model
        self.prompt     = prompt
        self.max_tokens = max_tokens
        self._enabled   = _LANGFUSE_ENABLED and _client is not None

    def update(self, response: Any) -> dict[str, int | float]:
        """
        Log token usage from an Anthropic response object to the current
        Langfuse generation span. Also returns a cost summary dict for
        memory/store.py logging.

        Args:
            response: anthropic.types.Message returned by client.messages.create()

        Returns:
            {input_tokens, output_tokens, cache_read_tokens,
             cache_write_tokens, estimated_cost_usd}
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}

        input_tokens        = getattr(usage, "input_tokens", 0)
        output_tokens       = getattr(usage, "output_tokens", 0)
        cache_read_tokens   = getattr(usage, "cache_read_input_tokens", 0)
        cache_write_tokens  = getattr(usage, "cache_creation_input_tokens", 0)

        # Billable input = uncached input only (cache reads are 10% price)
        billable_input = input_tokens - cache_read_tokens
        cost = (
            billable_input    * self._INPUT_COST_PER_M  / 1_000_000
            + output_tokens   * self._OUTPUT_COST_PER_M / 1_000_000
            + cache_read_tokens  * self._CACHE_READ_PER_M  / 1_000_000
            + cache_write_tokens * self._CACHE_WRITE_PER_M / 1_000_000
        )

        if self._enabled:
            try:
                _client.update_current_generation(
                    model=self.model,
                    usage={
                        "input":        input_tokens,
                        "output":       output_tokens,
                        "total":        input_tokens + output_tokens,
                        "unit":         "TOKENS",
                    },
                    metadata={
                        "cache_read_tokens":  cache_read_tokens,
                        "cache_write_tokens": cache_write_tokens,
                        "estimated_cost_usd": round(cost, 6),
                    },
                )
            except Exception as exc:      # noqa: BLE001
                logger.debug("Langfuse update_current_generation failed: %s", exc)

        return {
            "input_tokens":        input_tokens,
            "output_tokens":       output_tokens,
            "cache_read_tokens":   cache_read_tokens,
            "cache_write_tokens":  cache_write_tokens,
            "estimated_cost_usd":  round(cost, 6),
        }


@contextmanager
def trace_generation(
    name: str,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
) -> Generator[GenerationContext, None, None]:
    """
    Context manager wrapping a single Anthropic API call as a Langfuse
    generation span.

    Usage:
        with trace_generation("generate_sql", model, prompt) as gen:
            response = anthropic_client.messages.create(...)
            cost_info = gen.update(response)
    """
    ctx = GenerationContext(name, model, prompt, max_tokens)

    if _LANGFUSE_ENABLED and _client is not None:
        try:
            with _client.start_as_current_observation(
                type="generation",
                name=name,
                input=prompt,
                model=model,
            ):
                yield ctx
                return
        except Exception as exc:      # noqa: BLE001
            logger.debug("Langfuse start_as_current_observation failed: %s", exc)

    # No-op path
    yield ctx


def flush() -> None:
    """Flush all pending Langfuse spans. Call at end of each graph run."""
    if _LANGFUSE_ENABLED and _client is not None:
        try:
            _client.flush()
        except Exception as exc:      # noqa: BLE001
            logger.debug("Langfuse flush failed: %s", exc)
