"""
tests/test_org_api.py — Workspaces + members + shared resource access.
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import patch

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


def _auth_headers(client, *, prefix: str = "org") -> tuple[dict[str, str], str]:
    un = f"{prefix}_{uuid.uuid4().hex[:8]}"
    email = f"{un}@test.com"
    r = client.post(
        "/auth/register",
        json={"username": un, "email": email, "password": "Password1!"},
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    client.cookies.clear()
    return {"Authorization": f"Bearer {token}"}, email


def _patch_test_pg():
    return patch("api.routes.workspace._test_pg")


@pytest.fixture
def public_dns():
    with patch("backend.api.routes.runs.socket.gethostbyname", return_value="8.8.8.8"):
        with patch("api.routes.runs.socket.gethostbyname", return_value="8.8.8.8"):
            yield


class TestWorkspacesAPI:
    def test_register_bootstraps_personal_workspace(self, client):
        headers, _ = _auth_headers(client)
        r = client.get("/workspaces", headers=headers)
        assert r.status_code == 200, r.text
        workspaces = r.json()["workspaces"]
        assert len(workspaces) >= 1
        assert workspaces[0]["role"] == "owner"
        assert workspaces[0]["name"]

    def test_create_workspace(self, client):
        headers, _ = _auth_headers(client)
        r = client.post("/workspaces", headers=headers, json={"name": "Acme Analytics"})
        assert r.status_code == 201, r.text
        assert r.json()["name"] == "Acme Analytics"
        assert r.json()["role"] == "owner"

    def test_guest_has_no_workspaces(self, client):
        r = client.post("/auth/guest")
        assert r.status_code == 200
        token = r.json()["access_token"]
        client.cookies.clear()
        headers = {"Authorization": f"Bearer {token}"}
        listed = client.get("/workspaces", headers=headers).json()
        assert listed["workspaces"] == []

    def test_add_member_by_email_shares_connection(self, client, public_dns):
        owner_h, _ = _auth_headers(client, prefix="own")
        analyst_h, analyst_email = _auth_headers(client, prefix="an")

        ws = client.get("/workspaces", headers=owner_h).json()["workspaces"][0]
        ws_id = ws["workspace_id"]

        with _patch_test_pg():
            conn = client.post(
                "/connections",
                headers={**owner_h, "X-Workspace-Id": ws_id},
                json={
                    "name": "Team DB",
                    "host": "team.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            ).json()

        # Add analyst
        r = client.post(
            f"/workspaces/{ws_id}/members",
            headers=owner_h,
            json={"email": analyst_email, "role": "analyst"},
        )
        assert r.status_code == 201, r.text
        assert r.json()["role"] == "analyst"

        # Analyst can list/get via workspace header
        listed = client.get(
            "/connections",
            headers={**analyst_h, "X-Workspace-Id": ws_id},
        ).json()
        assert any(c["connection_id"] == conn["connection_id"] for c in listed["connections"])

        got = client.get(
            f"/connections/{conn['connection_id']}",
            headers=analyst_h,
        )
        assert got.status_code == 200

        # Analyst cannot delete
        denied = client.delete(
            f"/connections/{conn['connection_id']}",
            headers={**analyst_h, "X-Workspace-Id": ws_id},
        )
        assert denied.status_code == 403

        # Analyst cannot create
        with _patch_test_pg():
            create = client.post(
                "/connections",
                headers={**analyst_h, "X-Workspace-Id": ws_id},
                json={
                    "name": "Nope",
                    "host": "nope.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            )
        assert create.status_code == 403

    def test_foreign_workspace_header_forbidden(self, client):
        a, _ = _auth_headers(client, prefix="a")
        b, _ = _auth_headers(client, prefix="b")
        a_ws = client.get("/workspaces", headers=a).json()["workspaces"][0]["workspace_id"]
        r = client.get("/connections", headers={**b, "X-Workspace-Id": a_ws})
        assert r.status_code == 403
