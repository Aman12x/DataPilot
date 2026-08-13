"""
agents/analyze/stream_hub.py — per-run text-delta emitters.

Lets a graph node stream partial LLM output (the narrative draft) to whoever
is watching the run, without the agents layer importing the backend: the
backend registers an emitter for a run_id before invoking the graph, the node
calls emit(), and everything is a silent no-op when nobody registered —
tests, evals, and CLI invocations need no setup.

Emitters must never be able to break generation: emit() swallows every
exception the callback raises.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_emitters: dict[str, Callable[[dict], None]] = {}
_lock = threading.Lock()


def register(run_id: str, fn: Callable[[dict], None]) -> None:
    """Attach an emitter for this run. Payloads are event dicts, e.g.
    {"type": "narrative_delta", "text": "..."}."""
    if not run_id:
        return
    with _lock:
        _emitters[run_id] = fn


def unregister(run_id: str) -> None:
    with _lock:
        _emitters.pop(run_id, None)


def emit(run_id: str, payload: dict) -> None:
    """Deliver a payload to the run's emitter. No-op without one; a failing
    emitter is logged and dropped for the event, never raised into the node."""
    with _lock:
        fn = _emitters.get(run_id or "")
    if fn is None:
        return
    try:
        fn(payload)
    except Exception:
        logger.debug("stream_hub: emitter failed for run %s", run_id, exc_info=True)
