"""
tests/test_sse_resume_ids.py — every SSE event carries its stream id, and a
reconnect resumes from the id the client last saw.

The frontend opens a *new* EventSource on resume and on token refresh, so the
browser's automatic Last-Event-ID never applies; it passes ?last_id= instead.
Before this, a reconnect read the Redis stream from "$" and anything published
in the reconnect window (the narrative_start reset, the first deltas of a
revision) was dropped.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
_TESTS = os.path.dirname(__file__)
for p in (ROOT, BACKEND, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest_plugins = ["api_harness"]

from api.routes import runs as runs_route  # noqa: E402


def _auth(client):
    un = f"sse_{uuid.uuid4().hex[:8]}"
    r = client.post("/auth/register",
                    json={"username": un, "email": f"{un}@test.com", "password": "Password1!"})
    assert r.status_code == 201, r.text
    client.cookies.clear()
    return r.json()["access_token"]


def _run(client, access) -> str:
    r = client.post("/runs", json={"task": "analyse"}, headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    for _ in range(50):
        if run_id in client.app.state.graph._known_runs:
            break
        time.sleep(0.05)
    return run_id


def _stream(client, run_id, access, extra=""):
    r = client.get(f"/runs/{run_id}/stream-token", headers={"Authorization": f"Bearer {access}"})
    url = f"/runs/{run_id}/stream?stream_token={r.json()['stream_token']}{extra}"
    frames: list[tuple[str | None, dict]] = []
    cur_id: str | None = None
    with client.stream("GET", url) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            if raw.startswith("id: "):
                cur_id = raw[4:]
            elif raw.startswith("data: "):
                ev = json.loads(raw[6:])
                frames.append((cur_id, ev))
                cur_id = None
                if ev.get("type") in ("done", "error", "gate"):
                    break
    return frames


def _scripted_read_result(items):
    """Stand in for run_manager.read_result: hands out `items` in order,
    recording the last_id each call was made with."""
    calls: list[str] = []
    queue = list(items)

    async def _read(run_id, last_id):
        calls.append(last_id)
        return queue.pop(0) if queue else None

    return _read, calls


def test_every_event_carries_its_stream_id(client, monkeypatch):
    items = [
        {"type": "step", "label": "Writing SQL", "_stream_id": "100-0"},
        {"type": "narrative_delta", "text": "Treat", "_stream_id": "101-0"},
        {"ok": True, "_stream_id": "102-0"},
    ]
    read, calls = _scripted_read_result(items)
    monkeypatch.setattr(runs_route, "read_result", read)
    monkeypatch.setattr(runs_route, "is_invoke_in_flight", lambda rid: True)

    access = _auth(client)
    run_id = _run(client, access)
    frames = _stream(client, run_id, access)

    ids = [i for i, _ in frames]
    types = [e.get("type") for _, e in frames]
    assert types == ["step", "narrative_delta", "done"], frames
    assert ids == ["100-0", "101-0", "102-0"], ids
    # A fresh connection reads from "$" (only new entries).
    assert calls[0] == "$"


def test_reconnect_resumes_from_last_id(client, monkeypatch):
    items = [{"ok": True, "_stream_id": "205-0"}]
    read, calls = _scripted_read_result(items)
    monkeypatch.setattr(runs_route, "read_result", read)
    monkeypatch.setattr(runs_route, "is_invoke_in_flight", lambda rid: True)

    access = _auth(client)
    run_id = _run(client, access)
    _stream(client, run_id, access, extra="&last_id=204-0")
    assert calls[0] == "204-0"


def test_last_event_id_header_is_honoured(client, monkeypatch):
    """A browser auto-reconnect sends Last-Event-ID rather than ?last_id=."""
    items = [{"ok": True, "_stream_id": "305-0"}]
    read, calls = _scripted_read_result(items)
    monkeypatch.setattr(runs_route, "read_result", read)
    monkeypatch.setattr(runs_route, "is_invoke_in_flight", lambda rid: True)

    access = _auth(client)
    run_id = _run(client, access)
    r = client.get(f"/runs/{run_id}/stream-token", headers={"Authorization": f"Bearer {access}"})
    url = f"/runs/{run_id}/stream?stream_token={r.json()['stream_token']}"
    with client.stream("GET", url, headers={"Last-Event-ID": "304-0"}) as resp:
        for raw in resp.iter_lines():
            if raw.startswith("data: ") and json.loads(raw[6:]).get("type") == "done":
                break
    assert calls[0] == "304-0"


def test_garbage_last_id_falls_back_to_new_entries(client, monkeypatch):
    """Anything XREAD could not take must not 500 the stream."""
    items = [{"ok": True, "_stream_id": "405-0"}]
    read, calls = _scripted_read_result(items)
    monkeypatch.setattr(runs_route, "read_result", read)
    monkeypatch.setattr(runs_route, "is_invoke_in_flight", lambda rid: True)

    access = _auth(client)
    run_id = _run(client, access)
    _stream(client, run_id, access, extra="&last_id=not-an-id")
    assert calls[0] == "$"
