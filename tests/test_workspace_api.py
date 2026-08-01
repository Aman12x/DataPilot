"""
tests/test_workspace_api.py — API tests for connections + metric packs + start-run.

Uses the shared api_harness so FakeGraph / app.state stay consistent with test_api.py.
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import AsyncMock, patch

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

SAMPLE_CONFIG = {
    "primary_metric": "revenue",
    "metric_source_col": "revenue_usd",
    "metric_agg": "sum",
    "covariate": "prior_week_revenue",
    "metric_direction": "higher_is_better",
    "events_table": "transactions",
    "experiment_table": "assignments",
    "guardrail_metrics": ["refund_rate"],
    "segment_cols": ["country"],
}


def _auth_headers(client) -> dict[str, str]:
    """Register a user and return Bearer headers (cookies cleared)."""
    un = f"smb_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/auth/register",
        json={"username": un, "email": f"{un}@test.com", "password": "Password1!"},
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    client.cookies.clear()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth(client):
    return _auth_headers(client)


@pytest.fixture
def public_dns():
    with patch("backend.api.routes.runs.socket.gethostbyname", return_value="8.8.8.8"):
        with patch("api.routes.runs.socket.gethostbyname", return_value="8.8.8.8"):
            yield


def _patch_test_pg():
    return patch("api.routes.workspace._test_pg")


class TestConnectionsAPI:
    def test_create_without_live_test(self, client, auth, public_dns):
        with _patch_test_pg() as mock_test:
            mock_test.return_value = {
                "success": True, "error": None, "table_count": 3, "tables": ["a"],
            }
            r = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "Analytics",
                    "host": "db.example.com",
                    "port": 5432,
                    "dbname": "analytics",
                    "username": "reader",
                    "password": "secret",
                    "test": True,
                },
            )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["name"] == "Analytics"
        assert "password" not in data
        assert "password_enc" not in data
        assert data["last_test_ok"] is True

    def test_create_fails_on_bad_test(self, client, auth, public_dns):
        with _patch_test_pg() as mock_test:
            mock_test.return_value = {"success": False, "error": "timeout", "table_count": 0}
            r = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "Bad",
                    "host": "db.example.com",
                    "port": 5432,
                    "dbname": "x",
                    "username": "u",
                    "password": "p",
                    "test": True,
                },
            )
        assert r.status_code == 400
        assert "test failed" in r.json()["detail"].lower()

    def test_list_and_delete(self, client, auth, public_dns):
        with _patch_test_pg():
            created = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "C1",
                    "host": "analytics.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            ).json()

        listed = client.get("/connections", headers=auth).json()
        assert any(c["connection_id"] == created["connection_id"] for c in listed["connections"])

        r = client.delete(f"/connections/{created['connection_id']}", headers=auth)
        assert r.status_code == 204
        listed2 = client.get("/connections", headers=auth).json()
        assert all(c["connection_id"] != created["connection_id"] for c in listed2["connections"])

    def test_blocks_private_host(self, client, auth):
        r = client.post(
            "/connections",
            headers=auth,
            json={
                "name": "Internal",
                "host": "10.0.0.5",
                "port": 5432,
                "dbname": "d",
                "username": "u",
                "password": "p",
                "test": False,
            },
        )
        assert r.status_code == 400
        assert "not allowed" in r.json()["detail"].lower()

    def test_blocks_private_mysql_host(self, client, auth):
        r = client.post(
            "/connections",
            headers=auth,
            json={
                "name": "Internal MySQL",
                "backend": "mysql",
                "host": "192.168.1.10",
                "port": 3306,
                "dbname": "d",
                "username": "u",
                "password": "p",
                "test": False,
            },
        )
        assert r.status_code == 400
        assert "not allowed" in r.json()["detail"].lower()

    def test_create_mysql_connection(self, client, auth, public_dns):
        with _patch_test_pg() as mock_test:
            mock_test.return_value = {
                "success": True, "error": None, "table_count": 2, "tables": ["orders"],
            }
            r = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "MySQL Prod",
                    "backend": "mysql",
                    "host": "mysql.example.com",
                    "port": 3306,
                    "dbname": "shop",
                    "username": "reader",
                    "password": "secret",
                    "test": True,
                },
            )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["backend"] == "mysql"
        assert data["port"] == 3306
        assert "password" not in data

    def test_create_bigquery_connection(self, client, auth):
        creds = (
            '{"type":"service_account","client_email":"sa@proj.iam.gserviceaccount.com",'
            '"private_key":"-----BEGIN PRIVATE KEY-----\\nX\\n-----END PRIVATE KEY-----\\n"}'
        )
        with _patch_test_pg() as mock_test:
            mock_test.return_value = {
                "success": True, "error": None, "table_count": 1, "tables": ["events"],
            }
            r = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "BQ Analytics",
                    "backend": "bigquery",
                    "project_id": "my-gcp-project",
                    "dbname": "analytics",
                    "password": creds,
                    "test": True,
                },
            )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["backend"] == "bigquery"
        assert data["project_id"] == "my-gcp-project"
        assert data["dbname"] == "analytics"
        assert "password" not in data

    def test_bigquery_rejects_invalid_credentials_json(self, client, auth):
        r = client.post(
            "/connections",
            headers=auth,
            json={
                "name": "BQ Bad",
                "backend": "bigquery",
                "project_id": "p",
                "dbname": "d",
                "password": "not-json",
                "test": False,
            },
        )
        assert r.status_code == 400
        assert "json" in r.json()["detail"].lower()

    def test_ownership_isolation(self, client, public_dns):
        a = _auth_headers(client)
        b = _auth_headers(client)
        with _patch_test_pg():
            created = client.post(
                "/connections",
                headers=a,
                json={
                    "name": "A-only",
                    "host": "a.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            ).json()

        r = client.get(f"/connections/{created['connection_id']}", headers=b)
        assert r.status_code == 404
        r = client.delete(f"/connections/{created['connection_id']}", headers=b)
        assert r.status_code == 404


class TestMetricPacksAPI:
    def test_create_list_get(self, client, auth):
        r = client.post(
            "/metric-packs",
            headers=auth,
            json={
                "name": "Revenue",
                "description": "Core ecommerce metrics",
                "config": SAMPLE_CONFIG,
                "certified": True,
            },
        )
        assert r.status_code == 201, r.text
        pack = r.json()
        assert pack["certified"] is True
        assert pack["version"] == 1

        listed = client.get("/metric-packs", headers=auth).json()
        assert any(p["pack_id"] == pack["pack_id"] for p in listed["metric_packs"])

        got = client.get(f"/metric-packs/{pack['pack_id']}", headers=auth).json()
        assert got["config"]["primary_metric"] == "revenue"

    def test_invalid_config_400(self, client, auth):
        r = client.post(
            "/metric-packs",
            headers=auth,
            json={"name": "Bad", "config": {"primary_metric": "x"}},
        )
        assert r.status_code == 400


class TestAnnotationsAPI:
    def test_put_get_annotations(self, client, auth, public_dns):
        with _patch_test_pg():
            conn = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "Ann",
                    "host": "ann.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            ).json()

        r = client.put(
            f"/connections/{conn['connection_id']}/annotations",
            headers=auth,
            json={
                "annotations": {"events": {"revenue": "USD revenue"}},
                "synonyms": {"WAU": "weekly_active"},
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["annotations"]["events"]["revenue"] == "USD revenue"

        got = client.get(
            f"/connections/{conn['connection_id']}/annotations",
            headers=auth,
        ).json()
        assert got["synonyms"]["WAU"] == "weekly_active"

    def test_drift_without_snapshot(self, client, auth, public_dns):
        with _patch_test_pg():
            conn = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "D",
                    "host": "drift.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            ).json()
        r = client.get(
            f"/connections/{conn['connection_id']}/drift",
            headers=auth,
        )
        assert r.status_code == 200
        assert r.json()["has_snapshot"] is False


class TestStartRunWithPackAndConnection:
    def test_run_accepts_connection_and_pack(self, client, auth, public_dns):
        with _patch_test_pg():
            conn = client.post(
                "/connections",
                headers=auth,
                json={
                    "name": "C",
                    "host": "pg.example.com",
                    "port": 5432,
                    "dbname": "d",
                    "username": "u",
                    "password": "p",
                    "test": False,
                },
            ).json()

        pack = client.post(
            "/metric-packs",
            headers=auth,
            json={"name": "P", "config": SAMPLE_CONFIG, "certified": True},
        ).json()

        with patch("api.routes.runs.start_run", new_callable=AsyncMock) as mock_start:
            r = client.post(
                "/runs",
                headers=auth,
                json={
                    "task": "Did revenue increase?",
                    "db_backend": "postgres",
                    "connection_id": conn["connection_id"],
                    "metric_pack_id": pack["pack_id"],
                    "analysis_mode": "ab_test",
                },
            )
        assert r.status_code == 201, r.text
        assert mock_start.await_count == 1
        state = mock_start.await_args.args[2]
        assert state["connection_id"] == conn["connection_id"]
        assert state["metric_pack_certified"] is True
        assert state["metric_config"].primary_metric == "revenue"
        assert state["pg_host"] == "pg.example.com"
        assert state["pg_password"] == "p"
