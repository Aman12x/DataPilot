import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Ensure SECRET_KEY is set before any backend module is imported.
# backend/api/deps.py exits at import time if this is missing.
# Must satisfy deps.validate_secret_key (>=32 chars) so tests exercise the same
# policy production enforces.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-not-a-real-key")

import pytest


@pytest.fixture(autouse=True)
def _clear_auth_rate():
    """Reset the in-memory auth rate-limiter between tests so the 10/min cap
    doesn't cause false 429s when many tests call /register or /login."""
    # The test suite adds backend/ to sys.path so the module loads as
    # api.routes.auth; the running server loads it as backend.api.routes.auth.
    # Try both to ensure the correct in-process dict is cleared.
    for mod_path in ("api.routes.auth", "backend.api.routes.auth"):
        try:
            import importlib, sys
            mod = sys.modules.get(mod_path) or importlib.import_module(mod_path)
            mod._auth_rate.clear()
        except Exception:
            pass
    yield


@pytest.fixture(autouse=True)
def _reset_run_manager_globals():
    """Clear run_manager's process-wide run state *before* every test.

    run_manager keeps admission counters and run registries at module level, so
    they outlive the test that created them. If a test leaves `_active_invokes`
    incremented — a run whose `_release_slot` never fired because its event loop
    closed with the invoke task still pending — the counter stays leaked for the
    rest of the session. Once it reaches `_MAX_CONCURRENT`, `_invoke` rejects the
    next run and publishes {"ok": False, "error": "Server is busy..."} as the
    stream's first event instead of a step event, and whichever test reads that
    first event fails a long way from the cause.

    test_run_lifecycle.py has its own reset fixture, but it is teardown-only and
    file-local, so it protects that file's later tests and nothing else. This
    resets on setup, which is the direction contamination actually travels, and
    covers every file.

    Hardening, not a known-bug fix: no test is currently known to leak the
    counter, and this does not change the outcome of any test today.

    The shared graph executor is deliberately left alone — recreating a
    ThreadPoolExecutor per test is expensive, and the files that need a fresh one
    already shut it down themselves.
    """
    import sys

    mod = sys.modules.get("backend.api.run_manager") or sys.modules.get("api.run_manager")
    if mod is not None:
        mod._active_invokes = 0
        mod._active_by_scope.clear()
        mod._queues.clear()
        mod._cancel_events.clear()
        mod._active_tasks.clear()
    yield
