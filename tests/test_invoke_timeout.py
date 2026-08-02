"""
tests/test_invoke_timeout.py — a timed-out run must not leak its worker thread.

`asyncio.wait_for` cancels the coroutine that is waiting; it cannot stop a
thread that has already started. The timeout path used to give the admission
slot back immediately while the thread kept executing the rest of the graph.

`_MAX_CONCURRENT` sizes both the admission cap and the executor, so the next run
was admitted against a pool worker that was still busy — the executor queued it,
and a brand-new run hung with nothing wrong with it. On a small pool a couple of
timeouts was enough to stall the queue behind work nobody would ever read.

Two things fix it, and both are tested here: the slot is held until the thread
really exits, and the thread is told to stop at its next node boundary.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from agents import spend
from backend.api import run_manager


class _SteppingGraph:
    """A graph that yields node updates until told to stop.

    Each `stream()` iteration is one node boundary — the only place the worker
    can be interrupted. `hold` blocks *inside* a node, which is what makes the
    interesting window observable: a real node (an LLM call, a big pandas op)
    keeps running well past the moment the timeout is reported, and that window
    is exactly where the leaked slot used to be handed out.

    `spend_per_step` bills into the active meter so the post-timeout tail is a
    real, non-zero amount.
    """

    def __init__(self, *, steps: int = 10_000, spend_per_step: float = 0.0) -> None:
        self.steps = steps
        self.spend_per_step = spend_per_step
        self.entered = threading.Event()
        self.holding = threading.Event()
        self.hold = threading.Event()
        self.finished = threading.Event()
        self.steps_run = 0
        self.spent = 0.0
        self.reached_get_state = False

    def _spend(self) -> None:
        meter = spend.current_meter()
        if self.spend_per_step and meter is not None:
            meter.add(self.spend_per_step)
            self.spent += self.spend_per_step

    def stream(self, arg, config, stream_mode=None):
        self.entered.set()
        try:
            for i in range(self.steps):
                self.steps_run += 1
                self._spend()
                if i == 0:
                    # Stuck inside the first node until the test releases it.
                    self.holding.set()
                    self.hold.wait(timeout=10)
                    # Spent *after* the timeout was reported — the tail that
                    # the caller is no longer around to bill.
                    self._spend()
                else:
                    threading.Event().wait(0.005)
                yield {"unknown_node": {}}
        finally:
            self.finished.set()

    def get_state(self, config):
        self.reached_get_state = True
        return type("Snap", (), {"values": {}})()


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(run_manager, "_INVOKE_TIMEOUT", 0.3)
    yield
    run_manager._active_tasks.clear()
    run_manager._queues.clear()
    run_manager._cancel_events.clear()
    run_manager._active_invokes = 0
    if run_manager._graph_executor is not None:
        run_manager._graph_executor.shutdown(wait=False, cancel_futures=True)
        run_manager._graph_executor = None


async def _drain(seconds: float = 3.0, *, until) -> None:
    """Yield to the loop until `until()` is true or the budget runs out."""
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if until():
            return
        await asyncio.sleep(0.02)


def test_timed_out_worker_is_told_to_stop():
    """Without this the thread runs the whole remaining graph for nobody."""
    graph = _SteppingGraph()

    async def scenario():
        await run_manager.start_run(graph, "run-cancel", {}, "user-1")
        await asyncio.to_thread(graph.holding.wait, 5)
        await _drain(until=lambda: "run-cancel" in run_manager._run_errors)
        graph.hold.set()                       # the long node finishes
        await asyncio.to_thread(graph.finished.wait, 10)

    asyncio.run(scenario())
    assert graph.finished.is_set(), "worker never unwound after the timeout"
    # It stopped at the next boundary instead of running all 10k steps.
    assert graph.steps_run < 10, graph.steps_run
    assert not graph.reached_get_state


def test_admission_slot_is_held_until_the_worker_actually_exits():
    """The leak: the cap said "free" while a pool worker was still occupied."""
    graph = _SteppingGraph()
    observed: dict[str, int] = {}

    async def scenario():
        await run_manager.start_run(graph, "run-slot", {}, "user-1")
        await asyncio.to_thread(graph.holding.wait, 5)
        # The timeout has been reported, but the thread is still inside a node.
        await _drain(until=lambda: "run-slot" in run_manager._run_errors)
        observed["while_running"] = run_manager._active_invokes

        graph.hold.set()
        await asyncio.to_thread(graph.finished.wait, 10)
        await _drain(until=lambda: run_manager._active_invokes == 0)
        observed["after_exit"] = run_manager._active_invokes

    asyncio.run(scenario())
    assert observed["while_running"] == 1, (
        "slot was released while the worker thread was still running"
    )
    assert observed["after_exit"] == 0, "slot was never released"


def test_cancel_event_is_cleaned_up():
    """A leaked event per timed-out run is a slow memory leak of its own."""
    graph = _SteppingGraph()

    async def scenario():
        await run_manager.start_run(graph, "run-events", {}, "user-1")
        await asyncio.to_thread(graph.holding.wait, 5)
        await _drain(until=lambda: "run-events" in run_manager._run_errors)
        graph.hold.set()
        await asyncio.to_thread(graph.finished.wait, 10)
        await _drain(until=lambda: not run_manager._cancel_events)

    asyncio.run(scenario())
    assert run_manager._cancel_events == {}


def test_successful_run_releases_its_slot_and_event():
    graph = _SteppingGraph(steps=2)
    graph.hold.set()                            # nothing to hold up

    async def scenario():
        await run_manager.start_run(graph, "run-ok", {}, "user-1")
        await _drain(until=lambda: graph.reached_get_state)
        await _drain(until=lambda: run_manager._active_invokes == 0)

    asyncio.run(scenario())
    assert graph.reached_get_state
    assert run_manager._active_invokes == 0
    assert run_manager._cancel_events == {}


def test_shutdown_stops_worker_threads_at_a_node_boundary(monkeypatch):
    """Cancelling the coroutine alone left the thread running the whole graph."""
    monkeypatch.setattr(run_manager, "_INVOKE_TIMEOUT", 300)  # not a timeout test
    graph = _SteppingGraph()

    async def scenario():
        await run_manager.start_run(graph, "run-shutdown", {}, "user-1")
        await asyncio.to_thread(graph.holding.wait, 5)
        graph.hold.set()
        await asyncio.wait_for(run_manager.cancel_active_runs(), timeout=10)
        await asyncio.to_thread(graph.finished.wait, 10)

    asyncio.run(scenario())
    assert graph.finished.is_set(), "worker kept running after shutdown"
    assert graph.steps_run < 10_000


# ── Billing across the abandonment boundary ───────────────────────────────────

def test_meter_take_unbilled_drains():
    m = spend.Meter()
    m.add(1.0)
    m.add(0.5)
    assert m.take_unbilled() == (1.5, 2)
    assert m.take_unbilled() == (0.0, 0)
    m.add(0.25)
    assert m.take_unbilled() == (0.25, 1)
    # The running total is still the full amount — only the *billed* mark moves.
    assert m.total_usd == pytest.approx(1.75)
    assert m.calls == 3


def test_post_timeout_spend_is_billed_exactly_once(monkeypatch):
    """The tail costs real money, and charging it twice would too."""
    charges: list[tuple[str, float]] = []

    async def _record(scope, usd):
        charges.append((scope, usd))

    import backend.api.budget as budget
    monkeypatch.setattr(budget, "record_spend", _record)

    graph = _SteppingGraph(spend_per_step=0.01)

    async def scenario():
        await run_manager.start_run(graph, "run-bill", {}, "user-1")
        await asyncio.to_thread(graph.holding.wait, 5)
        # Bill #1 happens here, covering only the first node's spend.
        await _drain(until=lambda: len(charges) >= 1)
        graph.hold.set()
        await asyncio.to_thread(graph.finished.wait, 10)
        # Bill #2 covers whatever the abandoned thread spent on its way out.
        await _drain(until=lambda: len(charges) >= 2)

    asyncio.run(scenario())

    assert len(charges) == 2, f"expected a timeout bill and a tail bill, got {charges}"
    assert all(scope == "user:user-1" for scope, _ in charges)
    total_charged = sum(usd for _, usd in charges)
    assert total_charged == pytest.approx(graph.spent, rel=1e-6), (
        f"charged {total_charged} against {graph.spent} spent"
    )
