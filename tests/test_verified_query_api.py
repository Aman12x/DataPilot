"""API tests for the verified-query repository endpoints."""
from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
_TESTS = os.path.dirname(__file__)
for p in (ROOT, BACKEND, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest_plugins = ["api_harness"]


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    import memory.semantic_cache as sc

    def _embed(text: str) -> np.ndarray:
        vec = np.zeros(384, dtype=np.float32)
        vec[hash(text.split()[0].lower()) % 384] = 1.0
        return vec
    monkeypatch.setattr(sc, "embed", _embed)


@pytest.fixture(autouse=True)
def vq_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))


def _auth_headers(client) -> dict[str, str]:
    un = f"vq_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/auth/register",
        json={"username": un, "email": f"{un}@test.com", "password": "Password1!"},
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    client.cookies.clear()
    return {"Authorization": f"Bearer {token}"}


def _guest_headers(client) -> dict[str, str]:
    r = client.post("/auth/guest")
    assert r.status_code == 200, r.text
    token = r.json().get("access_token")
    client.cookies.clear()
    return {"Authorization": f"Bearer {token}"}


VALID = {
    "task": "Weekly revenue by product line",
    "sql": "SELECT week, product, SUM(revenue) FROM sales GROUP BY week, product",
    "name": "Weekly revenue",
}


def test_create_list_delete_roundtrip(client):
    auth = _auth_headers(client)
    r = client.post("/verified-queries", json=VALID, headers=auth)
    assert r.status_code == 201, r.text
    vq_id = r.json()["vq_id"]

    r = client.get("/verified-queries", headers=auth)
    rows = r.json()["verified_queries"]
    assert len(rows) == 1 and rows[0]["name"] == "Weekly revenue"
    assert rows[0]["source"] == "contributed"

    assert client.delete(f"/verified-queries/{vq_id}", headers=auth).status_code == 204
    assert client.get("/verified-queries", headers=auth).json()["verified_queries"] == []


def test_unsafe_sql_is_rejected(client):
    auth = _auth_headers(client)
    r = client.post("/verified-queries",
                    json={**VALID, "sql": "DROP TABLE sales"}, headers=auth)
    assert r.status_code == 400
    assert "Rejected SQL" in r.json()["detail"]


def test_guests_cannot_contribute(client):
    guest = _guest_headers(client)
    r = client.post("/verified-queries", json=VALID, headers=guest)
    assert r.status_code == 403


def test_users_do_not_see_each_others_queries(client):
    a = _auth_headers(client)
    b = _auth_headers(client)
    client.post("/verified-queries", json=VALID, headers=a)
    assert client.get("/verified-queries", headers=b).json()["verified_queries"] == []
    # And B cannot delete A's row.
    vq_id = client.get("/verified-queries", headers=a).json()["verified_queries"][0]["vq_id"]
    assert client.delete(f"/verified-queries/{vq_id}", headers=b).status_code == 404


def test_cap_is_enforced_with_a_clear_message(client, monkeypatch):
    from memory import verified_queries as vq_mod
    monkeypatch.setattr(vq_mod, "CONTRIBUTED_CAP", 1)
    auth = _auth_headers(client)
    assert client.post("/verified-queries", json=VALID, headers=auth).status_code == 201
    r = client.post("/verified-queries",
                    json={**VALID, "task": "Another question entirely"}, headers=auth)
    assert r.status_code == 400
    assert "limit" in r.json()["detail"].lower()
