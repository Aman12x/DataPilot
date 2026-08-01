"""
Auth routes must not block the event loop.

Two blocking things happened inline in `async def` handlers:

  - PBKDF2-HMAC-SHA256 at 260,000 iterations (~100-250ms of pure CPU) on
    register, login and password reset.
  - The Resend HTTP call. Contrary to an earlier note, resend's RequestsClient
    does carry a timeout (30s default), so this was bounded rather than
    unbounded -- but 30s of pinned loop on a --workers 1 backend is still an
    outage for every other request.
"""
import asyncio
import inspect
import time

import pytest

from backend.api import email as email_mod
from backend.api.routes import auth as auth_route


# ── Wiring ────────────────────────────────────────────────────────────────────


def test_password_hashing_is_offloaded():
    """PBKDF2 is the expensive part; it must not run on the loop."""
    for fn, call in [
        (auth_route.register, "create_user"),
        (auth_route.login, "verify_user"),
        (auth_route.reset_password, "update_password"),
    ]:
        src = inspect.getsource(fn)
        assert f"asyncio.to_thread(\n        {call}" in src or f"asyncio.to_thread({call}" in src, (
            f"{fn.__name__} still calls {call} on the event loop"
        )


def test_email_sends_use_the_async_wrappers():
    for fn in (auth_route.register, auth_route.login, auth_route.resend_verification):
        src = inspect.getsource(fn)
        if "send_verification_email" in src:
            assert "await send_verification_email_async(" in src, f"{fn.__name__} sends inline"
    src = inspect.getsource(auth_route.forgot_password)
    assert "await send_password_reset_async(" in src


def test_async_wrappers_exist_and_are_coroutines():
    assert inspect.iscoroutinefunction(email_mod.send_verification_email_async)
    assert inspect.iscoroutinefunction(email_mod.send_password_reset_async)
    # The sync originals stay: they are what runs inside the thread.
    assert not inspect.iscoroutinefunction(email_mod.send_verification_email)
    assert not inspect.iscoroutinefunction(email_mod.send_password_reset)


# ── Timeout configuration ─────────────────────────────────────────────────────


def test_email_timeout_is_shorter_than_the_sdk_default():
    assert email_mod._TIMEOUT < 30, "should tighten resend's 30s default"
    assert email_mod._TIMEOUT > 0


def test_configure_client_sets_the_timeout_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(email_mod, "_client_configured", False)
    resend = pytest.importorskip("resend")

    email_mod._configure_client()
    client = resend.default_http_client
    timeout = getattr(client, "_timeout", None)
    assert timeout == email_mod._TIMEOUT, f"timeout not applied (got {timeout})"

    # Second call is a no-op rather than rebuilding the client.
    before = resend.default_http_client
    email_mod._configure_client()
    assert resend.default_http_client is before


def test_configure_client_survives_a_missing_http_client_module(monkeypatch):
    """requirements allow resend>=2.0; the http_client module is not public API."""
    import builtins

    monkeypatch.setattr(email_mod, "_client_configured", False)
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if "http_client" in name:
            raise ImportError("no such module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    email_mod._configure_client()  # must not raise


# ── The loop stays responsive ─────────────────────────────────────────────────


def _heartbeat_during(coro_factory, seconds: float = 0.05):
    """Run a coroutine while ticking a heartbeat; return the tick count."""

    async def scenario():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                await asyncio.sleep(seconds)
                ticks += 1

        hb = asyncio.create_task(beat())
        await coro_factory()
        hb.cancel()
        return ticks

    return asyncio.run(scenario())


def test_loop_stays_free_during_a_slow_email_send(monkeypatch):
    def slow_send(to_email, token):
        time.sleep(1.0)  # stands in for a degraded provider

    monkeypatch.setattr(email_mod, "send_verification_email", slow_send)

    ticks = _heartbeat_during(
        lambda: email_mod.send_verification_email_async("a@b.com", "tok")
    )
    assert ticks > 5, f"event loop was blocked during send: {ticks} ticks"


def test_loop_stays_free_during_password_hashing():
    """A real PBKDF2 hash at the configured cost, off the loop."""
    from auth.store import _hash_password

    ticks = _heartbeat_during(
        lambda: asyncio.to_thread(_hash_password, "correct horse battery", "salt123")
    )
    assert ticks >= 1, f"event loop was blocked during hashing: {ticks} ticks"


def test_pbkdf2_cost_is_high_enough_to_matter():
    """Documents why this needed offloading at all."""
    from auth.store import _hash_password

    start = time.perf_counter()
    _hash_password("correct horse battery", "salt123")
    elapsed = time.perf_counter() - start
    assert elapsed > 0.02, (
        f"hash took only {elapsed*1000:.0f}ms -- if the iteration count dropped, "
        "that is a security regression worth checking"
    )
