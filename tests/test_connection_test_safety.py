"""
The connection-test endpoints reach out to a user-supplied host.

Three things made that dangerous on a --workers 1 backend:
  - _test_pg ran synchronously inside `async def`.
  - No driver carried a connect timeout, so a host that silently drops packets
    held the caller for the OS TCP timeout (minutes).
  - No workspace route had any rate limit, and /auth/guest mints identities
    freely, so the lever was available to anyone.
"""
import asyncio
import inspect
import socket
import time

import pytest

from backend.api.routes import workspace as ws
from tools import db_tools


# ── Driver timeouts ───────────────────────────────────────────────────────────


def test_connect_timeout_is_configured_and_short():
    assert 0 < db_tools.DB_CONNECT_TIMEOUT <= 30
    assert db_tools.DB_READ_TIMEOUT >= db_tools.DB_CONNECT_TIMEOUT


def test_postgres_and_mysql_pass_a_connect_timeout():
    pg = inspect.getsource(db_tools.DBConnection._query_postgres)
    assert "connect_timeout=DB_CONNECT_TIMEOUT" in pg
    my = inspect.getsource(db_tools.DBConnection._query_mysql)
    assert "connect_timeout=DB_CONNECT_TIMEOUT" in my
    assert "read_timeout=DB_READ_TIMEOUT" in my


def test_postgres_connect_actually_times_out():
    """Against a blackholed address the driver must give up quickly.

    198.51.100.0/24 is TEST-NET-2: reserved, unrouteable, drops silently.
    """
    psycopg2 = pytest.importorskip("psycopg2")
    db = db_tools.DBConnection(
        "postgres", host="198.51.100.1", port=5432,
        dbname="x", user="u", password="p",
    )
    start = time.perf_counter()
    with pytest.raises(Exception):
        db._query_postgres("SELECT 1")
    elapsed = time.perf_counter() - start
    assert elapsed < db_tools.DB_CONNECT_TIMEOUT + 8, (
        f"took {elapsed:.0f}s — connect_timeout did not apply"
    )


# ── Off-loop and bounded ──────────────────────────────────────────────────────


def test_routes_use_the_async_wrapper():
    for fn in (ws.create_connection, ws.test_saved_connection, ws.test_ephemeral):
        src = inspect.getsource(fn)
        assert "await _test_db_async(" in src, f"{fn.__name__} tests connections inline"
        assert "= _test_pg(" not in src


def test_async_wrapper_bounds_a_hanging_test(monkeypatch):
    """A driver that ignores its own timeout must not hold the request open.

    Timed inside the loop, not around asyncio.run(): run() waits for executor
    threads at shutdown, so it would measure the stuck thread rather than the
    request. That difference is real but only affects the worker thread --
    asyncio.to_thread cannot be cancelled, so the thread stays parked until the
    call returns. The request is released; the thread is not. Rate limiting is
    what bounds how many can pile up.
    """
    started = __import__("threading").Event()

    def never_returns(**kwargs):
        started.set()
        time.sleep(3)

    monkeypatch.setattr(ws, "_test_pg", never_returns)
    monkeypatch.setattr(ws, "_TEST_TIMEOUT", 0.3)

    async def scenario():
        start = time.perf_counter()
        result = await ws._test_db_async(backend="postgres", host="h")
        return result, time.perf_counter() - start

    async def main():
        res, elapsed = await scenario()
        return res, elapsed

    loop = asyncio.new_event_loop()
    try:
        result, elapsed = loop.run_until_complete(main())
    finally:
        loop.close()  # does not wait on the default executor

    assert started.is_set(), "the test never actually started"
    assert result["success"] is False
    assert "timed out" in result["error"]
    assert elapsed < 2, f"request was not released promptly ({elapsed:.1f}s)"


def test_event_loop_stays_free_during_a_slow_connection_test(monkeypatch):
    def slow(**kwargs):
        time.sleep(1.0)
        return {"success": True, "table_count": 0, "tables": []}

    monkeypatch.setattr(ws, "_test_pg", slow)
    monkeypatch.setattr(ws, "_TEST_TIMEOUT", 10)

    async def scenario():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        hb = asyncio.create_task(beat())
        await ws._test_db_async(backend="postgres", host="h")
        hb.cancel()
        return ticks

    ticks = asyncio.run(scenario())
    assert ticks > 5, f"event loop blocked during connection test: {ticks} ticks"


# ── Rate limiting ─────────────────────────────────────────────────────────────


def test_outbound_routes_are_rate_limited():
    """Every route that dials a user-chosen host must be throttled."""
    for fn in (ws.create_connection, ws.test_saved_connection, ws.test_ephemeral):
        src = inspect.getsource(fn)
        assert "check_auth_rate(request" in src, f"{fn.__name__} has no rate limit"


def test_rate_limited_routes_accept_a_request_object():
    """check_auth_rate keys on the client IP, so Request must be injected."""
    for fn in (ws.create_connection, ws.test_saved_connection, ws.test_ephemeral):
        assert "request" in inspect.signature(fn).parameters, fn.__name__


# ── SSRF guard still holds ────────────────────────────────────────────────────


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254"])
def test_private_hosts_are_still_rejected(host, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(ws, "_ALLOW_PRIVATE", False)
    with pytest.raises(HTTPException) as exc:
        ws._validate_host(host)
    assert exc.value.status_code in (400, 422)


def test_link_local_metadata_endpoint_is_blocked(monkeypatch):
    """169.254.169.254 is the cloud metadata service — the classic SSRF target."""
    from fastapi import HTTPException

    monkeypatch.setattr(ws, "_ALLOW_PRIVATE", False)
    with pytest.raises(HTTPException):
        ws._validate_host("169.254.169.254")
    # And via a hostname that resolves there, if resolution is attempted.
    try:
        socket.gethostbyname("localhost")
    except Exception:
        pytest.skip("no resolver")
    with pytest.raises(HTTPException):
        ws._validate_host("localhost")
