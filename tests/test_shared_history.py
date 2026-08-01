"""
tests/test_shared_history.py — Workspace-scoped run history + teammate read access.
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

from auth.workspace_store import init_workspace_tables  # noqa: E402

init_workspace_tables(os.environ.get("AUTH_DB_PATH"))


def _auth(client, prefix: str = "sh"):
    un = f"{prefix}_{uuid.uuid4().hex[:8]}"
    email = f"{un}@test.com"
    r = client.post(
        "/auth/register",
        json={"username": un, "email": email, "password": "Password1!"},
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    client.cookies.clear()
    return {"Authorization": f"Bearer {token}"}, email, un


class TestSharedHistoryAPI:
    def test_list_runs_includes_teammate_runs(self, client):
        owner_h, _, owner_un = _auth(client, "own")
        analyst_h, analyst_email, _ = _auth(client, "an")

        ws_id = client.get("/workspaces", headers=owner_h).json()["workspaces"][0][
            "workspace_id"
        ]
        r = client.post(
            f"/workspaces/{ws_id}/members",
            headers=owner_h,
            json={"email": analyst_email, "role": "analyst"},
        )
        assert r.status_code == 201, r.text

        # Seed memory store with a run owned by the workspace owner
        store = client.app.state.memory_store
        store.runs.append(
            {
                "run_id": str(uuid.uuid4()),
                "task": "Did checkout conversion lift?",
                "timestamp": "2026-08-01T00:00:00+00:00",
                "user_id": client.get("/auth/me", headers=owner_h).json()["user_id"],
                "username": owner_un,
                "workspace_id": ws_id,
                "analysis_mode": "ab_test",
                "eval_score": 0.9,
            }
        )

        listed = client.get(
            "/runs?limit=20",
            headers={**analyst_h, "X-Workspace-Id": ws_id},
        )
        assert listed.status_code == 200, listed.text
        runs = listed.json()
        assert any(r["task"] == "Did checkout conversion lift?" for r in runs)
        assert any(r.get("username") == owner_un for r in runs)

    def test_teammate_can_read_detail_but_not_resume(self, client):
        owner_h, _, _ = _auth(client, "own2")
        analyst_h, analyst_email, _ = _auth(client, "an2")

        ws_id = client.get("/workspaces", headers=owner_h).json()["workspaces"][0][
            "workspace_id"
        ]
        client.post(
            f"/workspaces/{ws_id}/members",
            headers=owner_h,
            json={"email": analyst_email, "role": "analyst"},
        )

        # Create a run through the API (FakeGraph records owner + workspace)
        create = client.post(
            "/runs",
            headers={**owner_h, "X-Workspace-Id": ws_id},
            json={"task": "Shared analysis", "db_backend": "duckdb", "analysis_mode": "general"},
        )
        assert create.status_code == 201, create.text
        run_id = create.json()["run_id"]

        # Wait briefly for async FakeGraph invoke to record the run
        for _ in range(50):
            graph = client.app.state.graph
            if run_id in graph._known_runs:
                break
            time.sleep(0.05)
        assert run_id in client.app.state.graph._known_runs
        assert client.app.state.graph._run_workspaces.get(run_id) == ws_id

        detail = client.get(f"/runs/{run_id}/detail", headers=analyst_h)
        assert detail.status_code == 200, detail.text
        assert detail.json().get("narrative") or detail.json().get("task")

        # Resume remains creator-only
        resume = client.post(
            f"/runs/{run_id}/resume",
            headers=analyst_h,
            json={"gate": "intent", "value": {"approved": True}},
        )
        assert resume.status_code == 403

    def test_outsider_cannot_list_or_read(self, client):
        owner_h, _, _ = _auth(client, "own3")
        outsider_h, _, _ = _auth(client, "out3")

        ws_id = client.get("/workspaces", headers=owner_h).json()["workspaces"][0][
            "workspace_id"
        ]
        create = client.post(
            "/runs",
            headers={**owner_h, "X-Workspace-Id": ws_id},
            json={"task": "Private to workspace", "db_backend": "duckdb"},
        )
        assert create.status_code == 201
        run_id = create.json()["run_id"]

        for _ in range(50):
            if run_id in client.app.state.graph._known_runs:
                break
            time.sleep(0.05)

        forbidden = client.get(
            "/runs",
            headers={**outsider_h, "X-Workspace-Id": ws_id},
        )
        assert forbidden.status_code == 403

        detail = client.get(f"/runs/{run_id}/detail", headers=outsider_h)
        assert detail.status_code == 403
