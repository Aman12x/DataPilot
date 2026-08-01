"""
tests/test_org_store.py — Workspaces + membership + resource scoping.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

from auth import org_store, workspace_store
from auth.store import create_user, init_db


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    path = str(tmp_path / "auth.db")
    monkeypatch.setenv("AUTH_DB_PATH", path)
    init_db(path)
    workspace_store.init_workspace_tables(path)
    org_store.init_org_tables(path)
    return path


def _user(path: str, name: str):
    u = create_user(name, f"{name}@test.com", "Password1!", path=path)
    assert not isinstance(u, str), u
    return u


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


class TestPersonalWorkspace:
    def test_ensure_idempotent(self, auth_db):
        u = _user(auth_db, "alice")
        w1 = org_store.ensure_personal_workspace(u.user_id, path=auth_db)
        w2 = org_store.ensure_personal_workspace(u.user_id, path=auth_db)
        assert w1.workspace_id == w2.workspace_id
        assert w1.role == "owner"

    def test_guests_rejected(self, auth_db):
        with pytest.raises(ValueError):
            org_store.ensure_personal_workspace("guest-abc", path=auth_db)


class TestMembership:
    def test_add_analyst_sees_shared_connection(self, auth_db):
        owner = _user(auth_db, "owner1")
        analyst = _user(auth_db, "analyst1")
        ws = org_store.ensure_personal_workspace(owner.user_id, path=auth_db)

        conn = workspace_store.create_connection(
            owner.user_id,
            name="Shared",
            host="db.example.com",
            port=5432,
            dbname="analytics",
            username="reader",
            password="secret",
            workspace_id=ws.workspace_id,
            path=auth_db,
        )
        org_store.add_member(
            ws.workspace_id, user_id=analyst.user_id, role="analyst", path=auth_db
        )

        # Analyst can read
        got = workspace_store.get_connection(analyst.user_id, conn.connection_id, path=auth_db)
        assert got is not None
        assert got.name == "Shared"

        listed = workspace_store.list_connections(
            analyst.user_id, workspace_id=ws.workspace_id, path=auth_db
        )
        assert len(listed) == 1

        # Analyst cannot mutate
        assert workspace_store.delete_connection(
            analyst.user_id, conn.connection_id, path=auth_db
        ) is False

        # Owner can mutate
        assert workspace_store.delete_connection(
            owner.user_id, conn.connection_id, path=auth_db
        ) is True

    def test_pack_workspace_scoping(self, auth_db):
        owner = _user(auth_db, "owner2")
        outsider = _user(auth_db, "out2")
        ws = org_store.ensure_personal_workspace(owner.user_id, path=auth_db)
        pack = workspace_store.create_metric_pack(
            owner.user_id,
            name="Rev",
            config=SAMPLE_CONFIG,
            workspace_id=ws.workspace_id,
            path=auth_db,
        )
        assert pack.workspace_id == ws.workspace_id
        assert workspace_store.get_metric_pack(outsider.user_id, pack.pack_id, path=auth_db) is None
        assert workspace_store.get_metric_pack(owner.user_id, pack.pack_id, path=auth_db) is not None

    def test_migrate_resources(self, auth_db):
        u = _user(auth_db, "mig")
        workspace_store.create_connection(
            u.user_id,
            name="Legacy",
            host="db.example.com",
            port=5432,
            dbname="x",
            username="u",
            password="p",
            path=auth_db,
        )
        workspace_store.create_metric_pack(
            u.user_id, name="P", config=SAMPLE_CONFIG, path=auth_db
        )
        ws = org_store.ensure_personal_workspace(u.user_id, path=auth_db)
        counts = org_store.migrate_user_resources_to_workspace(
            u.user_id, ws.workspace_id, path=auth_db
        )
        assert counts["connections"] == 1
        assert counts["packs"] == 1
        listed = workspace_store.list_connections(
            u.user_id, workspace_id=ws.workspace_id, path=auth_db
        )
        assert len(listed) == 1
        assert listed[0].workspace_id == ws.workspace_id

    def test_cannot_remove_last_owner(self, auth_db):
        u = _user(auth_db, "solo")
        ws = org_store.ensure_personal_workspace(u.user_id, path=auth_db)
        with pytest.raises(ValueError):
            org_store.remove_member(ws.workspace_id, u.user_id, path=auth_db)
