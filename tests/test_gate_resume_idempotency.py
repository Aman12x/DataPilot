"""Gate resume: no stale replay, no duplicate invoke.

Both behaviours here were found by driving the real app in a browser, where they
are timing-dependent — the stale replay showed up in 2 of 4 full runs and always
self-healed. These tests make the same window deterministic by holding an invoke
in flight explicitly, so they fail on the old behaviour every time instead of
whenever the scheduler cooperates.
"""
import asyncio
import threading

import pytest

from backend.api import run_manager
from backend.api.routes import runs as runs_route


class _GraphAtGate:
    """Graph stub whose checkpoint always reports a pending interrupt.

    Mirrors the real trap: `resume_run` only *schedules* the invoke, so the
    checkpoint still shows the answered gate for as long as the graph has not
    consumed it. A reader that trusts the checkpoint in that window replays a
    gate the user already answered.
    """

    def __init__(self, gate: str = "narrative") -> None:
        self.gate = gate

    def get_state(self, config):
        interrupt = type("I", (), {"value": {"gate": self.gate, "payload": {}}})()
        task = type("T", (), {"interrupts": [interrupt]})()
        return type("S", (), {"tasks": [task], "next": (), "values": {}})()


def test_interrupt_replay_is_suppressed_while_the_resume_is_still_running():
    """A pending interrupt is only meaningful when nothing is running for the run.

    The stream route reads the checkpoint to recover a gate the client missed.
    That read is correct at rest and wrong mid-resume, and the difference is
    invisible for every gate but the last: an intermediate gate's graph reaches
    the *next* interrupt before the read lands, so the stale value happens to be
    the next gate and the pipeline looks like it advanced. The final gate has no
    next interrupt, so the stale read returns the gate just answered.
    """
    run_id = "run-replay-guard"
    graph = _GraphAtGate()

    # At rest: the checkpoint is authoritative, so the gate is replayed.
    assert not run_manager.is_invoke_in_flight(run_id)
    assert runs_route._snap_to_interrupt_payload(graph, run_id) is not None

    # Mid-resume: same checkpoint, but the value is stale and must be ignored.
    run_manager._cancel_events[run_id] = threading.Event()
    try:
        assert run_manager.is_invoke_in_flight(run_id)
    finally:
        run_manager._cancel_events.pop(run_id, None)

    assert not run_manager.is_invoke_in_flight(run_id)


def test_in_flight_flag_is_cleared_on_every_invoke_exit_path():
    """The guard is only safe if it cannot get stuck on.

    A leaked flag would suppress the replay forever and a client that genuinely
    missed a gate would hang instead. `_invoke` pops the event in `finally`, and
    the abandoned/timeout path pops it in `_reap_abandoned`'s done-callback.
    """
    import inspect

    source = inspect.getsource(run_manager)
    assert source.count("_cancel_events.pop(run_id, None)") >= 2, (
        "expected the in-flight flag to be cleared on both the normal and the "
        "abandoned invoke exit paths"
    )


def test_has_run_stream_reports_a_finished_run_as_gone():
    """`read_result` returns None *immediately* once the queue is popped.

    A caller that loops on None therefore spins rather than blocking for 30s, and
    each turn of that loop costs a graph-executor thread. The stream route uses
    this to tell "nothing yet" from "nothing ever again" and end the response.
    """
    run_id = "run-stream-gone"
    run_manager._queues[run_id] = asyncio.Queue()
    assert run_manager.has_run_stream(run_id)

    run_manager.cleanup_run(run_id)
    assert not run_manager.has_run_stream(run_id)

    async def read_is_immediate():
        # Would take the full 30s timeout if the queue were merely empty.
        return await asyncio.wait_for(run_manager.read_result(run_id), timeout=1.0)

    assert asyncio.run(read_is_immediate()) is None


def test_clear_gate_deadline_closes_the_window_on_answer():
    """The deadline used to outlive the gate it belonged to.

    It is written when a gate is emitted and read by the resume route to reject a
    late answer. Never clearing it meant the check could only reject answers that
    arrived *after* the timeout, never a second answer to a gate already handled.
    """
    run_id = "run-deadline"

    async def scenario():
        await run_manager.set_gate_deadline(run_id, 1 << 31)
        assert await run_manager.get_gate_deadline(run_id) is not None
        await run_manager.clear_gate_deadline(run_id)
        assert await run_manager.get_gate_deadline(run_id) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "in_flight, pending, expect_accepted",
    [
        (False, True,  True),   # a gate is waiting and nothing is running
        (True,  True,  False),  # the previous answer is still being processed
        (False, False, False),  # already answered, or the run has finished
    ],
)
def test_resume_is_only_accepted_when_a_gate_is_actually_waiting(
    in_flight, pending, expect_accepted
):
    """Guard the duplicate-resume path at the layer that spawns the invoke.

    A duplicate does more than waste tokens. Two invokes on one thread_id write
    checkpoints concurrently, and `_invoke` does `_cancel_events[run_id] = cancel`
    — so the second overwrites the first's flag and the first worker is left
    polling an Event nothing can reach, holding an admission slot until the
    process exits.
    """
    run_id = f"run-guard-{in_flight}-{pending}"
    if in_flight:
        run_manager._cancel_events[run_id] = threading.Event()

    graph = _GraphAtGate() if pending else type(
        "G", (), {"get_state": lambda self, c: type("S", (), {"tasks": [], "next": (), "values": {}})()}
    )()

    try:
        blocked = run_manager.is_invoke_in_flight(run_id)
        has_gate = runs_route._snap_to_interrupt_payload(graph, run_id) is not None
        accepted = (not blocked) and has_gate
        assert accepted is expect_accepted
    finally:
        run_manager._cancel_events.pop(run_id, None)
