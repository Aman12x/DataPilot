"""
tests/test_connection_authz.py — who may rewrite a shared connection.

`_user_can_mutate_connection` granted write access on creator identity alone.
That is a permission handed out once and never re-checked, so demotion from
owner to analyst — a role documented as read-only for shared connections — and
removal from the workspace entirely both took nothing away. The creator could
still repoint the host and rotate the stored password on a connection the team
was still using.

Creation already requires owner (`workspace.py::_require_owner`), so requiring
it to mutate costs nothing to anyone whose role has not changed.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

from auth import org_store, workspace_store  # noqa: E402


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    path = str(tmp_path / "auth.db")
    monkeypatch.setenv("AUTH_DB_PATH", path)
    workspace_store.init_workspace_tables(path)
    org_store.init_org_tables(path)
    return path


def _workspace_connection(auth_db, creator="user-creator"):
    ws = org_store.create_workspace(creator, name="Analytics", path=auth_db)
    conn = workspace_store.create_connection(
        creator,
        name="warehouse",
        host="db.example.com",
        port=5432,
        dbname="analytics",
        username="svc",
        password="original-password",
        workspace_id=ws.workspace_id,
        path=auth_db,
    )
    return ws, conn


def _can_mutate(user_id, connection_id, auth_db):
    return workspace_store._user_can_mutate_connection(user_id, connection_id, path=auth_db)


# ── The regression ────────────────────────────────────────────────────────────

def test_creator_who_is_demoted_loses_write_access(auth_db):
    ws, conn = _workspace_connection(auth_db)
    # A second owner, so demoting the creator is allowed at all.
    org_store.add_member(ws.workspace_id, user_id="user-boss", role="owner", path=auth_db)
    assert _can_mutate("user-creator", conn.connection_id, auth_db)

    org_store.add_member(ws.workspace_id, user_id="user-creator", role="analyst", path=auth_db)
    assert not _can_mutate("user-creator", conn.connection_id, auth_db), (
        "demoted creator can still rewrite the connection"
    )


def test_creator_removed_from_the_workspace_loses_write_access(auth_db):
    ws, conn = _workspace_connection(auth_db)
    org_store.add_member(ws.workspace_id, user_id="user-boss", role="owner", path=auth_db)
    org_store.remove_member(ws.workspace_id, "user-creator", path=auth_db)
    assert not _can_mutate("user-creator", conn.connection_id, auth_db), (
        "removed member can still rewrite the connection"
    )


def test_demoted_creator_cannot_rotate_the_stored_password(auth_db):
    """The check is only worth anything if it actually stops the write."""
    ws, conn = _workspace_connection(auth_db)
    org_store.add_member(ws.workspace_id, user_id="user-boss", role="owner", path=auth_db)
    org_store.add_member(ws.workspace_id, user_id="user-creator", role="analyst", path=auth_db)

    result = workspace_store.update_connection(
        "user-creator", conn.connection_id, password="attacker-password", path=auth_db
    )
    assert result is None

    secrets = workspace_store.get_connection_secrets(
        "user-boss", conn.connection_id, path=auth_db
    )
    assert secrets is not None and secrets.password == "original-password"


# ── What must keep working ────────────────────────────────────────────────────

def test_workspace_owner_can_mutate(auth_db):
    ws, conn = _workspace_connection(auth_db)
    org_store.add_member(ws.workspace_id, user_id="user-boss", role="owner", path=auth_db)
    assert _can_mutate("user-boss", conn.connection_id, auth_db)


def test_creator_who_is_still_an_owner_can_mutate(auth_db):
    _, conn = _workspace_connection(auth_db)
    assert _can_mutate("user-creator", conn.connection_id, auth_db)
    updated = workspace_store.update_connection(
        "user-creator", conn.connection_id, host="db2.example.com", path=auth_db
    )
    assert updated is not None and updated.host == "db2.example.com"


def test_analyst_who_never_created_anything_cannot_mutate(auth_db):
    ws, conn = _workspace_connection(auth_db)
    org_store.add_member(ws.workspace_id, user_id="user-analyst", role="analyst", path=auth_db)
    assert not _can_mutate("user-analyst", conn.connection_id, auth_db)


def test_non_member_cannot_mutate(auth_db):
    _, conn = _workspace_connection(auth_db)
    assert not _can_mutate("user-stranger", conn.connection_id, auth_db)


def test_personal_connection_stays_creator_only(auth_db):
    """No workspace means no roles to consult — the creator is the whole ACL."""
    conn = workspace_store.create_connection(
        "user-solo",
        name="local",
        host="db.example.com",
        port=5432,
        dbname="analytics",
        username="svc",
        password="pw",
        path=auth_db,
    )
    assert _can_mutate("user-solo", conn.connection_id, auth_db)
    assert not _can_mutate("user-other", conn.connection_id, auth_db)


def test_unknown_connection_is_denied(auth_db):
    assert not _can_mutate("user-anyone", "00000000-0000-0000-0000-000000000000", auth_db)
