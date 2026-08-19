"""
tests/test_deck_route_spend.py — POST /runs/{id}/deck is metered and rate-limited.

5aa2aba moved deck generation out of the graph (off the approval path) and
into this route — which also moved it out from under run_manager's spend
meter. The route must price its own LLM call, bill the caller's budget
scope, and refuse to re-run a failing generation without bound.
"""

from __future__ import annotations

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

from agents import spend  # noqa: E402
from api import run_manager  # noqa: E402
from api.routes import runs as runs_route  # noqa: E402


def _auth(client):
    un = f"deck_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/auth/register",
        json={"username": un, "email": f"{un}@test.com", "password": "Password1!"},
    )
    assert r.status_code == 201, r.text
    client.cookies.clear()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _finished_run(client, headers) -> str:
    create = client.post(
        "/runs", headers=headers,
        json={"task": "deck me", "db_backend": "duckdb", "analysis_mode": "general"},
    )
    assert create.status_code == 201, create.text
    run_id = create.json()["run_id"]
    for _ in range(50):
        if run_id in client.app.state.graph._known_runs:
            break
        time.sleep(0.05)
    assert run_id in client.app.state.graph._known_runs
    return run_id


@pytest.fixture
def approved_state(client, monkeypatch):
    """FakeGraph's state has no final_narrative; give it one, no deck yet."""
    graph = client.app.state.graph
    orig = graph.get_state

    def _get_state(config, **kw):
        st = orig(config, **kw)
        st.values = {**st.values, "final_narrative": "# Approved\n\nShip it."}
        return st

    monkeypatch.setattr(graph, "get_state", _get_state)
    monkeypatch.setattr(graph, "update_state", lambda *a, **k: None, raising=False)
    yield graph


class TestDeckSpend:
    def test_generation_is_metered_and_billed_to_caller(self, client, approved_state, monkeypatch):
        billed: list[tuple[str, float]] = []

        async def _record(scope, usd):
            billed.append((scope, usd))

        def _fake_deck(values, narrative):
            m = spend.current_meter()
            assert m is not None, "deck generation ran with no active spend meter"
            m.add(0.0123)
            return {"slides": [{"title": "t"}]}

        monkeypatch.setattr(runs_route, "record_spend", _record)
        import agents.analyze.nodes_narrative as nn
        monkeypatch.setattr(nn, "_generate_deck", _fake_deck)

        headers = _auth(client)
        run_id = _finished_run(client, headers)
        r = client.post(f"/runs/{run_id}/deck", headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["deck_data"] == {"slides": [{"title": "t"}]}

        assert len(billed) == 1
        scope, usd = billed[0]
        assert scope.startswith("user:")
        assert usd == pytest.approx(0.0123)

    def test_failed_generation_is_rate_limited(self, client, approved_state, monkeypatch):
        """A failed deck is not persisted; without a bucket every retry pays."""
        import agents.analyze.nodes_narrative as nn
        monkeypatch.setattr(nn, "_generate_deck", lambda v, n: {})
        monkeypatch.setattr(run_manager, "_MAX_RESUMES", 2)

        headers = _auth(client)
        run_id = _finished_run(client, headers)
        codes = [client.post(f"/runs/{run_id}/deck", headers=headers).status_code
                 for _ in range(3)]
        assert codes[:2] == [200, 200], codes
        assert codes[2] == 429, codes
