"""
tests/test_event_loop_liveness.py — a slow store call must not freeze the server.

`tests/test_event_loop_blocking.py` proves no blocking call *appears* inside an
`async def`. That is a static claim about the source. This one is the behavioural
counterpart: with a store call artificially slowed, a second, unrelated request
must still be served while it runs.

Worth having both. The scan can be satisfied by a call that was moved off the
loop in form but not in effect, and a future refactor could reintroduce blocking
through a path the scan does not model — a sync helper called from an async
route was exactly how six of these hid in the first place.

`/auth/forgot-password` is the probe because it reaches a store function
(`create_reset_token`) without needing an authenticated session, and it answers
202 either way, so nothing depends on the user existing.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import uuid

import httpx
import pytest

_TESTS = os.path.dirname(__file__)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

pytest_plugins = ["api_harness"]

_BLOCK_SECONDS = 0.6
# Generous: the point is "served concurrently", not a latency benchmark. On a
# blocked loop /health cannot answer before _BLOCK_SECONDS no matter the margin.
_HEALTH_BUDGET = _BLOCK_SECONDS / 2


@pytest.fixture
def slow_store(monkeypatch):
    """Make one store lookup take real wall-clock time and record its timeline."""
    import api.routes.auth as auth_routes

    seen: dict = {"entered": threading.Event()}

    def _slow_create_reset_token(email: str):
        seen["thread"] = threading.get_ident()
        seen["entered"].set()
        time.sleep(_BLOCK_SECONDS)
        seen["exited_at"] = time.perf_counter()
        return ""      # falsy: no email is sent, so the probe stays self-contained

    monkeypatch.setattr(auth_routes, "create_reset_token", _slow_create_reset_token)
    return seen


def _run(coro):
    return asyncio.run(coro)


def test_a_slow_store_call_does_not_stall_an_unrelated_request(slow_store):
    from api.main import app
    from api_harness import FakeGraph, FakeMemoryStore

    async def scenario():
        # The lifespan is not run by ASGITransport, so wire up what the routes
        # read off app.state themselves.
        app.state.graph = FakeGraph()
        app.state.memory_store = FakeMemoryStore()
        loop_thread = threading.get_ident()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            slow = asyncio.create_task(
                ac.post("/auth/forgot-password", json={"email": f"{uuid.uuid4().hex}@t.com"})
            )
            # Spin on the loop until the slow call is known to be in flight.
            # Deliberately not `await asyncio.to_thread(event.wait)`: if the loop
            # were blocked, that would only resume *after* the block finished and
            # the comparison below would be vacuous. Spinning here means a blocked
            # loop cannot get past this line until the block is over, which is
            # precisely what makes the assertion detect it.
            deadline = time.perf_counter() + 5
            while not slow_store["entered"].is_set() and time.perf_counter() < deadline:
                await asyncio.sleep(0)

            health = await ac.get("/health")
            health_done_at = time.perf_counter()
            slow_response = await slow
        return loop_thread, health, health_done_at, slow_response

    loop_thread, health, health_done_at, slow_response = _run(scenario())

    assert health.status_code == 200
    assert slow_response.status_code == 202
    # Behaviour first: /health finished before the slow call did, so the two
    # genuinely overlapped. A blocked loop cannot produce this ordering.
    assert health_done_at < slow_store["exited_at"], (
        f"/health only completed {health_done_at - slow_store['exited_at']:.2f}s "
        f"*after* the {_BLOCK_SECONDS}s store call finished — the loop was blocked"
    )
    # Then the mechanism, which says *why* — a thread, not a lucky schedule.
    assert slow_store["thread"] != loop_thread, "the store call ran on the event loop thread"


def test_the_probe_would_catch_a_blocking_call(slow_store):
    """Guard for the guard: sleeping on the loop must fail the same assertions.

    Without this, a probe that silently stopped reaching the slowed function
    would report a fast /health and look like a pass.
    """
    from api.main import app
    from api_harness import FakeGraph, FakeMemoryStore

    async def scenario():
        app.state.graph = FakeGraph()
        app.state.memory_store = FakeMemoryStore()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            # Block the loop directly, the way an un-offloaded store call would.
            started = time.perf_counter()
            blocker = asyncio.create_task(_block_the_loop())
            await ac.get("/health")
            elapsed = time.perf_counter() - started
            await blocker
        return elapsed

    async def _block_the_loop():
        time.sleep(_BLOCK_SECONDS)

    elapsed = _run(scenario())
    assert elapsed >= _HEALTH_BUDGET, (
        f"/health answered in {elapsed:.2f}s despite a blocked loop — "
        "this probe cannot detect blocking and the test above proves nothing"
    )
