"""
agents/spend.py — per-run LLM spend metering.

Four of the seven `messages.create()` call sites in agents/analyze never folded
their cost into AgentState, so run cost was under-reported by more than half.
Metering happens at the client wrapper in node_shared instead, which no new call
site can bypass by forgetting to merge a dict.

The active meter is a ContextVar. Each graph run executes on its own worker
thread, and every thread carries its own context, so concurrent runs accumulate
independently without locking across runs.
"""
from __future__ import annotations

import contextvars
import logging
import threading
from contextlib import contextmanager
from typing import Any, Generator

from agents import pricing

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar["Meter | None"] = contextvars.ContextVar(
    "datapilot_spend_meter", default=None
)


class Meter:
    """Accumulates the USD cost of the LLM calls made during one graph run."""

    def __init__(self) -> None:
        self.total_usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.total_usd += usd
            self.calls += 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Meter ${self.total_usd:.6f} over {self.calls} call(s)>"


@contextmanager
def meter(existing: "Meter | None" = None) -> Generator[Meter, None, None]:
    """Collect the spend of everything that runs inside this block.

    Pass an `existing` meter when the caller needs to read the total even if the
    block never returns — a timed-out run still spent the tokens it spent.
    """
    m = existing if existing is not None else Meter()
    token = _current.set(m)
    try:
        yield m
    finally:
        _current.reset(token)


def current_meter() -> Meter | None:
    return _current.get()


def record(model: str, response: Any) -> float:
    """Price one API response and add it to the active meter.

    Returns the cost so callers can also report it per-call. Never raises: a
    metering failure must not abort an analysis that already succeeded.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        cost = pricing.cost_from_usage(model, usage)
    except Exception:
        logger.debug("spend.record failed for model %s", model, exc_info=True)
        return 0.0

    m = _current.get()
    if m is not None:
        m.add(cost)
    return cost
