"""
auth/org_store.py — Workspaces + membership (Phase 3).

Roles:
  owner   — manage connections, packs, members
  analyst — run analyses; read shared connections/packs/history

Guests stay outside workspaces (demo path). Real users get a personal
workspace on first ensure_personal_workspace() call.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from auth.store import _auth_db_path, _connect, init_db as _init_auth_db

logger = logging.getLogger(__name__)

Role = Literal["owner", "analyst"]
_ROLE_RANK = {"analyst": 1, "owner": 2}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_org_tables(path: str | None = None) -> None:
    path = path or _auth_db_path()
    _init_auth_db(path)
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id   TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                created_by     TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                deleted_at     TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS workspace_members (
                workspace_id   TEXT NOT NULL,
                user_id        TEXT NOT NULL,
                role           TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                PRIMARY KEY (workspace_id, user_id)
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_workspace_members_user "
            "ON workspace_members(user_id)"
        )


@dataclass
class WorkspacePublic:
    workspace_id: str
    name: str
    role: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "role": self.role,
            "created_at": self.created_at,
        }


@dataclass
class MemberPublic:
    user_id: str
    username: str
    email: str
    role: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at,
        }


def ensure_personal_workspace(
    user_id: str,
    *,
    name: str | None = None,
    path: str | None = None,
) -> WorkspacePublic:
    """
    Return the user's personal workspace, creating one if needed.

    Idempotent. Guests (user_id starting with guest-) are rejected.
    """
    if not user_id or user_id.startswith("guest-"):
        raise ValueError("Guests cannot own workspaces")

    path = path or _auth_db_path()
    init_org_tables(path)

    with _connect(path) as con:
        row = con.execute(
            """
            SELECT w.workspace_id, w.name, w.created_at, m.role
            FROM workspaces w
            JOIN workspace_members m ON m.workspace_id = w.workspace_id
            WHERE m.user_id = ? AND w.deleted_at IS NULL
            ORDER BY w.created_at ASC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    if row:
        d = dict(row)
        return WorkspacePublic(
            workspace_id=d["workspace_id"],
            name=d["name"],
            role=d["role"],
            created_at=d["created_at"],
        )

    workspace_id = str(uuid.uuid4())
    now = _utcnow()
    ws_name = (name or "Personal").strip() or "Personal"
    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO workspaces (workspace_id, name, created_by, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (workspace_id, ws_name, user_id, now),
        )
        con.execute(
            """
            INSERT INTO workspace_members (workspace_id, user_id, role, created_at)
            VALUES (?, ?, 'owner', ?)
            """,
            (workspace_id, user_id, now),
        )
    logger.info("workspace.created personal user=%s id=%s", user_id, workspace_id)
    return WorkspacePublic(
        workspace_id=workspace_id, name=ws_name, role="owner", created_at=now
    )


def list_workspaces(user_id: str, path: str | None = None) -> list[WorkspacePublic]:
    path = path or _auth_db_path()
    init_org_tables(path)
    with _connect(path) as con:
        rows = con.execute(
            """
            SELECT w.workspace_id, w.name, w.created_at, m.role
            FROM workspaces w
            JOIN workspace_members m ON m.workspace_id = w.workspace_id
            WHERE m.user_id = ? AND w.deleted_at IS NULL
            ORDER BY w.created_at ASC
            """,
            (user_id,),
        ).fetchall()
    return [
        WorkspacePublic(
            workspace_id=dict(r)["workspace_id"],
            name=dict(r)["name"],
            role=dict(r)["role"],
            created_at=dict(r)["created_at"],
        )
        for r in rows
    ]


def create_workspace(
    user_id: str,
    *,
    name: str,
    path: str | None = None,
) -> WorkspacePublic:
    if not user_id or user_id.startswith("guest-"):
        raise ValueError("Guests cannot create workspaces")
    path = path or _auth_db_path()
    init_org_tables(path)
    workspace_id = str(uuid.uuid4())
    now = _utcnow()
    ws_name = name.strip()
    if not ws_name:
        raise ValueError("Workspace name required")
    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO workspaces (workspace_id, name, created_by, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (workspace_id, ws_name, user_id, now),
        )
        con.execute(
            """
            INSERT INTO workspace_members (workspace_id, user_id, role, created_at)
            VALUES (?, ?, 'owner', ?)
            """,
            (workspace_id, user_id, now),
        )
    return WorkspacePublic(
        workspace_id=workspace_id, name=ws_name, role="owner", created_at=now
    )


def get_membership(
    user_id: str, workspace_id: str, path: str | None = None
) -> str | None:
    """Return role string or None if not a member."""
    path = path or _auth_db_path()
    init_org_tables(path)
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT m.role FROM workspace_members m
            JOIN workspaces w ON w.workspace_id = m.workspace_id
            WHERE m.workspace_id = ? AND m.user_id = ? AND w.deleted_at IS NULL
            """,
            (workspace_id, user_id),
        ).fetchone()
    return dict(row)["role"] if row else None


def require_role(
    user_id: str,
    workspace_id: str,
    *,
    min_role: Role = "analyst",
    path: str | None = None,
) -> str:
    """
    Ensure user is a member with at least min_role.
    Returns the user's role. Raises PermissionError on failure.
    """
    role = get_membership(user_id, workspace_id, path=path)
    if role is None:
        raise PermissionError("Not a member of this workspace")
    if _ROLE_RANK.get(role, 0) < _ROLE_RANK.get(min_role, 0):
        raise PermissionError(f"Requires {min_role} role (have {role})")
    return role


def list_members(
    workspace_id: str, path: str | None = None
) -> list[MemberPublic]:
    path = path or _auth_db_path()
    init_org_tables(path)
    with _connect(path) as con:
        rows = con.execute(
            """
            SELECT m.user_id, m.role, m.created_at,
                   COALESCE(u.username, '') AS username,
                   COALESCE(u.email, '') AS email
            FROM workspace_members m
            LEFT JOIN users u ON u.user_id = m.user_id
            WHERE m.workspace_id = ?
            ORDER BY m.created_at ASC
            """,
            (workspace_id,),
        ).fetchall()
    return [
        MemberPublic(
            user_id=dict(r)["user_id"],
            username=dict(r)["username"],
            email=dict(r)["email"],
            role=dict(r)["role"],
            created_at=dict(r)["created_at"],
        )
        for r in rows
    ]


def add_member(
    workspace_id: str,
    *,
    user_id: str,
    role: Role = "analyst",
    path: str | None = None,
) -> MemberPublic:
    if role not in _ROLE_RANK:
        raise ValueError(f"Invalid role: {role}")
    path = path or _auth_db_path()
    init_org_tables(path)
    now = _utcnow()
    with _connect(path) as con:
        existing = con.execute(
            "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        ).fetchone()
        if existing:
            con.execute(
                "UPDATE workspace_members SET role = ? WHERE workspace_id = ? AND user_id = ?",
                (role, workspace_id, user_id),
            )
        else:
            con.execute(
                """
                INSERT INTO workspace_members (workspace_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (workspace_id, user_id, role, now),
            )
    members = [m for m in list_members(workspace_id, path=path) if m.user_id == user_id]
    return members[0]


def remove_member(
    workspace_id: str, user_id: str, path: str | None = None
) -> bool:
    path = path or _auth_db_path()
    init_org_tables(path)
    with _connect(path) as con:
        # Don't remove the last owner
        owners = con.execute(
            """
            SELECT user_id FROM workspace_members
            WHERE workspace_id = ? AND role = 'owner'
            """,
            (workspace_id,),
        ).fetchall()
        owner_ids = {dict(r)["user_id"] for r in owners}
        if user_id in owner_ids and len(owner_ids) <= 1:
            raise ValueError("Cannot remove the last owner")
        cur = con.execute(
            "DELETE FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        try:
            return int(cur.rowcount or 0) > 0
        except Exception:
            return True


def migrate_user_resources_to_workspace(
    user_id: str, workspace_id: str, path: str | None = None
) -> dict[str, int]:
    """
    Backfill workspace_id on connections/packs still owned only by user_id.
    Safe to re-run.
    """
    path = path or _auth_db_path()
    init_org_tables(path)
    # Ensure columns exist (workspace_store.init adds them)
    from auth.workspace_store import init_workspace_tables
    init_workspace_tables(path)

    counts = {"connections": 0, "packs": 0}
    with _connect(path) as con:
        cur = con.execute(
            """
            UPDATE db_connections
            SET workspace_id = ?
            WHERE user_id = ? AND (workspace_id IS NULL OR workspace_id = '')
              AND deleted_at IS NULL
            """,
            (workspace_id, user_id),
        )
        try:
            counts["connections"] = int(cur.rowcount or 0)
        except Exception:
            pass
        cur = con.execute(
            """
            UPDATE metric_packs
            SET workspace_id = ?
            WHERE user_id = ? AND (workspace_id IS NULL OR workspace_id = '')
              AND deleted_at IS NULL
            """,
            (workspace_id, user_id),
        )
        try:
            counts["packs"] = int(cur.rowcount or 0)
        except Exception:
            pass
    return counts
