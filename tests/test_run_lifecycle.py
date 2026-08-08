"""
Regression tests for shutdown cleanup and the Railway volume mount path.

Both cover deploy-breaking bugs:
  1. backend/api/main.py imported run_manager.cancel_active_runs, which did not
     exist — every graceful shutdown raised ImportError before the sweeper tasks
     were cancelled or Redis was closed.
  2. railway.toml documented a volume mount at /app/memory, which shadows the
     `memory/` Python package and breaks startup (see DEVLOG.md).
"""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.api import run_manager


class _BlockingGraph:
    """Stands in for a compiled LangGraph whose stream() blocks in a worker thread."""

    def __init__(self, released: threading.Event) -> None:
        self.released = released
        self.entered = threading.Event()
        self.thread_name = ""

    def stream(self, arg, config, stream_mode=None):
        self.thread_name = threading.current_thread().name
        self.entered.set()
        self.released.wait(timeout=5)
        return iter(())

    def get_state(self, config):
        raise AssertionError("cancelled run should never reach get_state")


@pytest.fixture(autouse=True)
def _reset_run_manager():
    """Keep module-level run state from leaking between tests."""
    yield
    run_manager._active_tasks.clear()
    run_manager._queues.clear()
    run_manager._active_invokes = 0
    if run_manager._graph_executor is not None:
        run_manager._graph_executor.shutdown(wait=False, cancel_futures=True)
        run_manager._graph_executor = None


def test_cancel_active_runs_is_importable():
    """main.py's shutdown path imports this symbol by name."""
    from backend.api.run_manager import cancel_active_runs

    assert callable(cancel_active_runs)


def test_lifespan_shutdown_imports_resolve():
    """Every name main.py imports from run_manager at shutdown must exist."""
    source = Path(__file__).resolve().parents[1] / "backend" / "api" / "main.py"
    text = source.read_text()
    assert "from .run_manager import cancel_active_runs" in text
    assert hasattr(run_manager, "cancel_active_runs")


def test_cancel_active_runs_noop_when_idle():
    asyncio.run(run_manager.cancel_active_runs())


def test_cancel_active_runs_cancels_in_flight_task():
    released = threading.Event()
    graph = _BlockingGraph(released)

    async def scenario():
        await run_manager.start_run(graph, "run-cancel", {}, "user-1")
        # Let the task reach the worker thread before cancelling.
        await asyncio.wait_for(asyncio.to_thread(graph.entered.wait, 5), timeout=6)
        assert len(run_manager._active_tasks) == 1

        await asyncio.wait_for(run_manager.cancel_active_runs(), timeout=5)

        assert run_manager._active_tasks == set()
        assert run_manager._active_invokes == 0

    try:
        asyncio.run(scenario())
    finally:
        released.set()


def test_invoke_tasks_are_strongly_referenced():
    """asyncio holds only weak refs; start_run must keep its own."""
    released = threading.Event()
    graph = _BlockingGraph(released)

    async def scenario():
        await run_manager.start_run(graph, "run-ref", {}, "user-1")
        assert len(run_manager._active_tasks) == 1
        released.set()
        await asyncio.wait_for(run_manager.cancel_active_runs(), timeout=5)

    try:
        asyncio.run(scenario())
    finally:
        released.set()


# ── Thread-pool isolation ─────────────────────────────────────────────────────


def test_graph_runs_on_dedicated_executor():
    """Graph work must not share asyncio's process-wide default executor."""
    released = threading.Event()
    graph = _BlockingGraph(released)

    async def scenario():
        await run_manager.start_run(graph, "run-exec", {}, "user-1")
        await asyncio.wait_for(asyncio.to_thread(graph.entered.wait, 5), timeout=6)
        assert graph.thread_name.startswith("graph-invoke"), graph.thread_name
        released.set()
        await asyncio.wait_for(run_manager.cancel_active_runs(), timeout=5)

    try:
        asyncio.run(scenario())
    finally:
        released.set()


def test_executor_is_sized_to_the_concurrency_cap():
    """The admission cap is only honest if the pool can actually serve it."""
    executor = run_manager._get_graph_executor()
    assert executor._max_workers == run_manager._MAX_CONCURRENT


def test_sse_reads_do_not_consume_worker_threads():
    """A waiting SSE reader must hold zero threads.

    Previously each connected client blocked a pool thread for a 30s poll, so a
    handful of viewers could occupy every slot and stall all graph execution.
    """
    async def scenario():
        run_manager._queues["run-sse"] = asyncio.Queue()
        before = threading.active_count()

        reader = asyncio.create_task(run_manager.read_result("run-sse"))
        await asyncio.sleep(0.2)
        assert threading.active_count() == before, "SSE read spawned a thread"

        await run_manager._publish_result("run-sse", {"type": "step", "label": "ok"})
        result = await asyncio.wait_for(reader, timeout=5)
        assert result["label"] == "ok"

    asyncio.run(scenario())


def test_step_events_stream_from_worker_thread_to_reader():
    """End-to-end: graph.stream() in a worker thread → SSE reader on the loop."""

    class _SteppingGraph:
        def stream(self, arg, config, stream_mode=None):
            yield {"execute_query": {"query_result": [1, 2, 3]}}

        def get_state(self, config):
            return type("S", (), {"values": {"done": True}})()

    async def scenario():
        await run_manager.start_run(_SteppingGraph(), "run-e2e", {}, "user-1")

        step = await asyncio.wait_for(run_manager.read_result("run-e2e"), timeout=5)
        # Name the event we actually got. Every non-step event on this stream is
        # an {"ok": ...} dict with no "type" key, so asserting the key directly
        # raises a bare KeyError that cannot distinguish a rejected run from a
        # failed one from an out-of-order one. Keeping the message is what
        # identified the real cause: it reported
        #     expected a step event first, got {'ok': True, 'snap': {'done': True}}
        # which ruled out rejection and failure (the run clearly succeeded) and
        # pointed at the fire-and-forget step publish fixed in _publish_sync.
        # test_step_publish_is_synchronous_... now pins that ordering directly;
        # this assertion stays as the end-to-end backstop.
        assert "type" in step, f"expected a step event first, got {step!r}"
        assert step["type"] == "step"
        assert step["node"] == "execute_query"
        assert step["detail"] == "3 rows returned"

        final = await asyncio.wait_for(run_manager.read_result("run-e2e"), timeout=5)
        assert final["ok"] is True
        assert final["snap"] == {"done": True}

    asyncio.run(scenario())


def test_worker_thread_publish_reaches_the_queue():
    """_publish_sync runs off-loop; it must use the loop it was handed."""
    async def scenario():
        loop = asyncio.get_running_loop()
        run_manager._queues["run-pub"] = asyncio.Queue()

        def worker():
            run_manager._publish_sync("run-pub", {"type": "step", "label": "from-thread"}, loop)

        await asyncio.to_thread(worker)
        result = await asyncio.wait_for(run_manager.read_result("run-pub"), timeout=5)
        assert result["label"] == "from-thread"

    asyncio.run(scenario())


def test_step_publish_is_synchronous_so_the_final_result_cannot_overtake_it(monkeypatch):
    """_publish_sync must not return until the payload is really on the stream.

    The in-memory path used to be fire-and-forget
    (`loop.call_soon_threadsafe(q.put_nowait, payload)`), so the worker thread
    returned while the step event was still only *scheduled* as a loop callback.
    _invoke then published the run's final {"ok": True} and a reader could see
    the run finish before its last chain-of-thought step. CI caught exactly that:

        expected a step event first, got {'ok': True, 'snap': {'done': True}}

    Checking the queue depth as _publish_sync returns is not enough: an idle
    loop drains a call_soon_threadsafe callback so promptly that the old code
    passed that check most of the time. Instead make the publish observably
    slow, so "did _publish_sync wait for it?" has one answer and no race.
    Fire-and-forget never reaches _publish_result at all on this path, so the
    flag stays clear and this fails deterministically.
    """
    published = threading.Event()
    real_publish = run_manager._publish_result

    async def slow_publish(run_id, payload):
        await asyncio.sleep(0.2)
        await real_publish(run_id, payload)
        published.set()

    monkeypatch.setattr(run_manager, "_publish_result", slow_publish)

    async def scenario():
        loop = asyncio.get_running_loop()
        run_manager._queues["run-sync"] = asyncio.Queue()

        def worker():
            run_manager._publish_sync("run-sync", {"type": "step", "label": "x"}, loop)
            return published.is_set()

        assert await asyncio.to_thread(worker), (
            "_publish_sync returned before the step reached the stream; "
            "the run's final result can overtake it"
        )

    asyncio.run(scenario())


def test_redis_mode_publishes_step_events_from_worker_thread(monkeypatch):
    """Redis mode silently dropped every step event.

    _publish_sync called asyncio.get_event_loop() from a worker thread, which
    raises when that thread has no loop; _stream_graph swallowed the error, so
    enabling Redis stripped the entire chain-of-thought stream.
    """
    published: list[tuple[str, dict]] = []

    class _FakeRedis:
        async def xadd(self, key, fields):
            published.append((key, fields))

        async def expire(self, key, ttl):
            pass

    monkeypatch.setattr(run_manager, "_redis", _FakeRedis())

    async def scenario():
        loop = asyncio.get_running_loop()

        def worker():
            run_manager._publish_sync("run-redis", {"type": "step", "label": "ok"}, loop)

        await asyncio.to_thread(worker)

    asyncio.run(scenario())

    assert published, "step event never reached Redis"
    key, fields = published[0]
    assert key == "run:run-redis"
    assert json.loads(fields["data"])["label"] == "ok"


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point every persistent path at tmp_path and force single-pod mode."""
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "datapilot_memory.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path


@pytest.fixture
def stub_background_init():
    """Skip the ~30s sample-data build and MiniLM warm-up during lifespan tests."""
    with patch("runpy.run_path"), patch("backend.api.main._prewarm_embedder"):
        yield


def test_real_lifespan_starts_and_shuts_down(isolated_env, stub_background_init):
    """The production lifespan must complete BOTH phases without raising.

    tests/test_api.py replaces app.router.lifespan_context with a stub so route
    tests don't boot the real graph. That stub is fine, but it means the real
    startup/shutdown path had no coverage at all — which is how an ImportError
    on every single shutdown shipped undetected.
    """
    from backend.api.main import app, lifespan

    async def scenario():
        async with lifespan(app):
            assert app.state.graph is not None
            assert app.state.memory_store is not None
        return True

    assert asyncio.run(scenario()) is True


def test_lifespan_honours_db_path_env_vars(isolated_env, stub_background_init):
    """railway.toml's volume fix depends on these env vars actually redirecting."""
    from backend.api.main import app, lifespan

    async def scenario():
        async with lifespan(app):
            pass

    asyncio.run(scenario())

    assert (isolated_env / "graph.db").exists(), "GRAPH_DB_PATH ignored"
    assert (isolated_env / "auth.db").exists(), "AUTH_DB_PATH ignored"
    assert (isolated_env / "uploads").is_dir(), "UPLOAD_DIR ignored"


def test_lifespan_shutdown_cancels_background_tasks(isolated_env, stub_background_init):
    """The sweeper and warm-up tasks must not outlive the app."""
    from backend.api.main import app, lifespan

    async def scenario():
        async with lifespan(app):
            before = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
            assert before, "expected background tasks during startup"
        await asyncio.sleep(0)
        return {t for t in before if not t.done()}

    assert asyncio.run(scenario()) == set()


def test_railway_volume_does_not_shadow_memory_package():
    root = Path(__file__).resolve().parents[1]
    toml = (root / "railway.toml").read_text()

    assert (root / "memory" / "__init__.py").exists(), "memory/ is a Python package"
    assert "/app/db" in toml
    # The mount path must never be the package directory. The string may still
    # appear in the explanatory warning, so check only directive-style lines.
    for line in toml.splitlines():
        stripped = line.lstrip("# ").strip()
        if stripped.startswith("Mount path:") or stripped.startswith("Railway CLI:"):
            assert "/app/memory" not in stripped, f"volume shadows memory/: {line}"


def test_railway_documents_db_path_env_vars():
    """The mount persists nothing unless these point into it."""
    toml = (Path(__file__).resolve().parents[1] / "railway.toml").read_text()
    for var in ("GRAPH_DB_PATH", "AUTH_DB_PATH", "MEMORY_DB_PATH", "UPLOAD_DIR"):
        assert f"{var}=/app/db" in toml, f"{var} not pointed at the volume"
