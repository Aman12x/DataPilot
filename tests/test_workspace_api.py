"""
tests/test_workspace_api.py — API tests for connections + metric packs + start-run.

Uses the same FastAPI TestClient + fake-lifespan pattern as test_api.py so
startup does not build a real LangGraph / generate DuckDB data.
"""

from __future__ import annotations

import os
import sys
import types
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
for p in (ROOT, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _stub_if_missing(name: str, **attrs):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _stub(name, **attrs)


_stub_if_missing("anthropic")
_stub_if_missing("langfuse")
_stub_if_missing("langfuse.decorators", observe=lambda **kw: (lambda f: f))
_stub_if_missing("sentence_transformers")

if not hasattr(sys.modules.get("langfuse.decorators", object()), "observe"):
    sys.modules["langfuse.decorators"] = _stub(
        "langfuse.decorators", observe=lambda **kw: (lambda f: f)
    )


@asynccontextmanager
async def _test_lifespan(app):
    from api.run_manager import set_redis_client

    set_redis_client(None)
    app.state.graph = MagicMock()
    app.state.memory_store = MagicMock()
    yield


_AUTH_DB = f"/tmp/test_ws_auth_{uuid.uuid4().hex}.db"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256"
os.environ["AUTH_DB_PATH"] = _AUTH_DB
os.environ.setdefault("MEMORY_DB_PATH", f"/tmp/test_ws_mem_{uuid.uuid4().hex}.db")
os.environ.setdefault("UPLOAD_DIR", f"/tmp/test_ws_uploads_{uuid.uuid4().hex}")
os.environ.setdefault("GRAPH_DB_PATH", f"/tmp/test_ws_graph_{uuid.uuid4().hex}.db")
os.environ["AUTH_AUTO_VERIFY_EMAIL"] = "true"
os.environ["AUTH_RETURN_TOKENS"] = "true"
os.environ["AUTH_RATE_MAX_ATTEMPTS"] = "10000"
os.environ["ALLOW_PRIVATE_DB_HOSTS"] = "false"

# Reset Fernet cache so SECRET_KEY above is used
import backend.api.crypto_secrets as _crypto  # noqa: E402

_crypto._FERNET = None

from auth.workspace_store import init_workspace_tables  # noqa: E402

init_workspace_tables(_AUTH_DB)

from api.main import app  # noqa: E402

app.router.lifespan_context = _test_lifespan  # type: ignore[assignment]

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


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _auth_headers(client) -> dict[str, str]:
    """Register a user and return Bearer headers.

    Clears cookies so HttpOnly auth cookies from register don't override the
    Authorization header (cookie wins in get_current_user).
    """
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
    """Force host validation to a public IP (avoids real DNS / NXDOMAIN)."""
    # Dual import paths (api.* vs backend.api.*) exist in this repo — patch both.
    with patch("backend.api.routes.runs.socket.gethostbyname", return_value="8.8.8.8"):
        with patch("api.routes.runs.socket.gethostbyname", return_value="8.8.8.8"):
            yield


def _patch_test_pg():
    """Patch connection tester on whichever module object the router uses."""
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
        args = mock_start.await_args.args
        state = args[2]
        assert state["connection_id"] == conn["connection_id"]
        assert state["metric_pack_certified"] is True
        assert state["metric_config"].primary_metric == "revenue"
        assert state["pg_host"] == "pg.example.com"
        # Password resolved for first load_schema; durable key is connection_id
        assert state["pg_password"] == "p"
