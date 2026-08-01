"""
auth/workspace_store.py — Saved DB connections + metric packs per user.

Industry-grade requirements:
  - Connection passwords encrypted at rest (Fernet)
  - Ownership checks on every read/update/delete
  - Never return decrypted passwords to API consumers
  - Soft-delete for audit trail
  - Metric packs versioned; certified flag skips Metric Config Gate
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from auth.store import _auth_db_path, _connect, init_db as _init_auth_db

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_workspace_tables(path: str | None = None) -> None:
    """Ensure connection + metric pack tables exist (idempotent)."""
    path = path or _auth_db_path()
    _init_auth_db(path)
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS db_connections (
                connection_id   TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                name            TEXT NOT NULL,
                backend         TEXT NOT NULL DEFAULT 'postgres',
                host            TEXT NOT NULL,
                port            INTEGER NOT NULL DEFAULT 5432,
                dbname          TEXT NOT NULL,
                username        TEXT NOT NULL,
                password_enc    TEXT NOT NULL,
                sslmode         TEXT NOT NULL DEFAULT 'prefer',
                last_tested_at  TEXT,
                last_test_ok    INTEGER,
                last_test_error TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                deleted_at      TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS metric_packs (
                pack_id         TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                name            TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                config_json     TEXT NOT NULL,
                certified       INTEGER NOT NULL DEFAULT 0,
                connection_id   TEXT,
                version         INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                deleted_at      TEXT
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_db_connections_user "
            "ON db_connections(user_id)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_metric_packs_user "
            "ON metric_packs(user_id)"
        )
        con.execute("""
            CREATE TABLE IF NOT EXISTS schema_annotations (
                connection_id   TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                annotations_json TEXT NOT NULL,
                synonyms_json   TEXT NOT NULL DEFAULT '{}',
                updated_at      TEXT NOT NULL
            )
        """)
        # Schema snapshot + workspace scoping columns (idempotent ALTERs)
        for col, defn in (
            ("schema_snapshot_json", "TEXT"),
            ("schema_hash", "TEXT"),
            ("schema_snapshot_at", "TEXT"),
            ("workspace_id", "TEXT"),
            ("project_id", "TEXT"),
        ):
            try:
                con.execute(f"ALTER TABLE db_connections ADD COLUMN {col} {defn}")
            except Exception:
                pass  # already exists
        try:
            con.execute("ALTER TABLE metric_packs ADD COLUMN workspace_id TEXT")
        except Exception:
            pass
        try:
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_db_connections_workspace "
                "ON db_connections(workspace_id)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_metric_packs_workspace "
                "ON metric_packs(workspace_id)"
            )
        except Exception:
            pass
        # Org tables (workspaces / members)
        try:
            from auth.org_store import init_org_tables
            init_org_tables(path)
        except Exception:
            pass


# ── Connection DTOs ───────────────────────────────────────────────────────────

@dataclass
class ConnectionPublic:
    connection_id: str
    name: str
    backend: str
    host: str
    port: int
    dbname: str
    username: str
    sslmode: str
    last_tested_at: str | None
    last_test_ok: bool | None
    last_test_error: str | None
    created_at: str
    updated_at: str
    workspace_id: str | None = None
    project_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "name": self.name,
            "backend": self.backend,
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "username": self.username,
            "sslmode": self.sslmode,
            "last_tested_at": self.last_tested_at,
            "last_test_ok": self.last_test_ok,
            "last_test_error": self.last_test_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
        }


@dataclass
class ConnectionSecrets:
    """Internal — never serialise to API responses."""
    connection_id: str
    user_id: str
    backend: str
    host: str
    port: int
    dbname: str
    username: str
    password: str
    sslmode: str
    project_id: str = ""


def _row_to_public(row: Any) -> ConnectionPublic:
    d = dict(row)
    ok = d.get("last_test_ok")
    return ConnectionPublic(
        connection_id=d["connection_id"],
        name=d["name"],
        backend=d["backend"],
        host=d["host"],
        port=int(d["port"] or 0),
        dbname=d["dbname"],
        username=d["username"],
        sslmode=d.get("sslmode") or "prefer",
        last_tested_at=d.get("last_tested_at"),
        last_test_ok=None if ok is None else bool(ok),
        last_test_error=d.get("last_test_error"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        workspace_id=d.get("workspace_id") or None,
        project_id=d.get("project_id") or None,
    )


def _user_can_access_connection(
    user_id: str, row: Any, path: str | None = None
) -> bool:
    """Owner user_id OR workspace membership."""
    d = dict(row)
    if d.get("user_id") == user_id:
        return True
    ws = d.get("workspace_id") or ""
    if not ws:
        return False
    from auth.org_store import get_membership
    return get_membership(user_id, ws, path=path) is not None


def create_connection(
    user_id: str,
    *,
    name: str,
    host: str = "",
    port: int = 0,
    dbname: str = "",
    username: str = "",
    password: str = "",
    backend: str = "postgres",
    sslmode: str = "prefer",
    project_id: str = "",
    workspace_id: str | None = None,
    path: str | None = None,
) -> ConnectionPublic:
    from backend.api.crypto_secrets import encrypt_secret

    path = path or _auth_db_path()
    init_workspace_tables(path)
    connection_id = str(uuid.uuid4())
    now = _utcnow()
    password_enc = encrypt_secret(password)

    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO db_connections (
                connection_id, user_id, name, backend, host, port, dbname,
                username, password_enc, sslmode, created_at, updated_at,
                workspace_id, project_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                connection_id, user_id, name.strip(), backend, (host or "").strip(),
                int(port or 0), (dbname or "").strip(), (username or "").strip(),
                password_enc, sslmode or "prefer", now, now, workspace_id,
                (project_id or "").strip() or None,
            ),
        )
    return get_connection(user_id, connection_id, path=path)  # type: ignore[return-value]


def list_connections(
    user_id: str,
    path: str | None = None,
    workspace_id: str | None = None,
) -> list[ConnectionPublic]:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    with _connect(path) as con:
        if workspace_id:
            rows = con.execute(
                """
                SELECT * FROM db_connections
                WHERE workspace_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC
                """,
                (workspace_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT * FROM db_connections
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
    return [_row_to_public(r) for r in rows]


def get_connection(
    user_id: str, connection_id: str, path: str | None = None
) -> ConnectionPublic | None:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT * FROM db_connections
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (connection_id,),
        ).fetchone()
    if not row or not _user_can_access_connection(user_id, row, path=path):
        return None
    return _row_to_public(row)


def get_connection_secrets(
    user_id: str, connection_id: str, path: str | None = None
) -> ConnectionSecrets | None:
    """Resolve decrypted credentials for query execution. Internal only."""
    from backend.api.crypto_secrets import decrypt_secret

    path = path or _auth_db_path()
    init_workspace_tables(path)
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT * FROM db_connections
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (connection_id,),
        ).fetchone()
    if not row or not _user_can_access_connection(user_id, row, path=path):
        return None
    d = dict(row)
    return ConnectionSecrets(
        connection_id=d["connection_id"],
        user_id=d["user_id"],
        backend=d["backend"],
        host=d["host"] or "",
        port=int(d["port"] or 0),
        dbname=d["dbname"] or "",
        username=d["username"] or "",
        password=decrypt_secret(d["password_enc"]),
        sslmode=d.get("sslmode") or "prefer",
        project_id=d.get("project_id") or "",
    )


def _user_can_mutate_connection(
    user_id: str, connection_id: str, path: str | None = None
) -> bool:
    """Creator or workspace owner can mutate connection config/secrets."""
    path = path or _auth_db_path()
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT user_id, workspace_id FROM db_connections
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (connection_id,),
        ).fetchone()
    if not row:
        return False
    d = dict(row)
    if d.get("user_id") == user_id:
        return True
    ws = d.get("workspace_id") or ""
    if not ws:
        return False
    from auth.org_store import get_membership
    return get_membership(user_id, ws, path=path) == "owner"


def update_connection(
    user_id: str,
    connection_id: str,
    *,
    name: str | None = None,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    username: str | None = None,
    password: str | None = None,
    sslmode: str | None = None,
    path: str | None = None,
) -> ConnectionPublic | None:
    from backend.api.crypto_secrets import encrypt_secret

    path = path or _auth_db_path()
    existing = get_connection(user_id, connection_id, path=path)
    if not existing:
        return None
    if not _user_can_mutate_connection(user_id, connection_id, path=path):
        return None

    now = _utcnow()
    fields: list[str] = []
    params: list[Any] = []

    def _set(col: str, val: Any) -> None:
        fields.append(f"{col} = ?")
        params.append(val)

    if name is not None:
        _set("name", name.strip())
    if host is not None:
        _set("host", host.strip())
    if port is not None:
        _set("port", int(port))
    if dbname is not None:
        _set("dbname", dbname.strip())
    if username is not None:
        _set("username", username.strip())
    if password is not None and password != "":
        _set("password_enc", encrypt_secret(password))
    if sslmode is not None:
        _set("sslmode", sslmode)

    if not fields:
        return existing

    _set("updated_at", now)
    params.append(connection_id)

    with _connect(path) as con:
        con.execute(
            f"UPDATE db_connections SET {', '.join(fields)} "
            "WHERE connection_id = ? AND deleted_at IS NULL",
            tuple(params),
        )
    return get_connection(user_id, connection_id, path=path)


def delete_connection(user_id: str, connection_id: str, path: str | None = None) -> bool:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not _user_can_mutate_connection(user_id, connection_id, path=path):
        return False
    now = _utcnow()
    with _connect(path) as con:
        cur = con.execute(
            """
            UPDATE db_connections SET deleted_at = ?, updated_at = ?
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (now, now, connection_id),
        )
        try:
            return int(cur.rowcount or 0) > 0
        except Exception:
            row = con.execute(
                """
                SELECT 1 FROM db_connections
                WHERE connection_id = ? AND deleted_at IS NOT NULL
                """,
                (connection_id,),
            ).fetchone()
            return row is not None


def record_connection_test(
    user_id: str,
    connection_id: str,
    *,
    ok: bool,
    error: str | None = None,
    path: str | None = None,
) -> None:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not get_connection(user_id, connection_id, path=path):
        return
    now = _utcnow()
    with _connect(path) as con:
        con.execute(
            """
            UPDATE db_connections
            SET last_tested_at = ?, last_test_ok = ?, last_test_error = ?, updated_at = ?
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (now, 1 if ok else 0, (error or "")[:500], now, connection_id),
        )


# ── Metric packs ──────────────────────────────────────────────────────────────

@dataclass
class MetricPackPublic:
    pack_id: str
    name: str
    description: str
    config: dict[str, Any]
    certified: bool
    connection_id: str | None
    version: int
    created_at: str
    updated_at: str
    workspace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "certified": self.certified,
            "connection_id": self.connection_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace_id": self.workspace_id,
        }


def _pack_from_row(row: Any) -> MetricPackPublic:
    d = dict(row)
    return MetricPackPublic(
        pack_id=d["pack_id"],
        name=d["name"],
        description=d.get("description") or "",
        config=json.loads(d["config_json"]),
        certified=bool(d.get("certified")),
        connection_id=d.get("connection_id") or None,
        version=int(d.get("version") or 1),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        workspace_id=d.get("workspace_id") or None,
    )


def _user_can_access_pack(user_id: str, row: Any, path: str | None = None) -> bool:
    d = dict(row)
    if d.get("user_id") == user_id:
        return True
    ws = d.get("workspace_id") or ""
    if not ws:
        return False
    from auth.org_store import get_membership
    return get_membership(user_id, ws, path=path) is not None


def _user_can_mutate_pack(user_id: str, pack_id: str, path: str | None = None) -> bool:
    path = path or _auth_db_path()
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT user_id, workspace_id FROM metric_packs
            WHERE pack_id = ? AND deleted_at IS NULL
            """,
            (pack_id,),
        ).fetchone()
    if not row:
        return False
    d = dict(row)
    if d.get("user_id") == user_id:
        return True
    ws = d.get("workspace_id") or ""
    if not ws:
        return False
    from auth.org_store import get_membership
    return get_membership(user_id, ws, path=path) == "owner"


def create_metric_pack(
    user_id: str,
    *,
    name: str,
    config: dict[str, Any],
    description: str = "",
    certified: bool = False,
    connection_id: str | None = None,
    workspace_id: str | None = None,
    path: str | None = None,
) -> MetricPackPublic:
    from config.analysis_config import MetricConfig

    # Validate against MetricConfig schema before persist
    MetricConfig(**config)

    path = path or _auth_db_path()
    init_workspace_tables(path)
    pack_id = str(uuid.uuid4())
    now = _utcnow()

    if connection_id:
        if not get_connection(user_id, connection_id, path=path):
            raise ValueError("connection_id not found or not accessible")

    with _connect(path) as con:
        con.execute(
            """
            INSERT INTO metric_packs (
                pack_id, user_id, workspace_id, name, description, config_json,
                certified, connection_id, version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                pack_id, user_id, workspace_id, name.strip(), description.strip(),
                json.dumps(config), 1 if certified else 0,
                connection_id, now, now,
            ),
        )
    return get_metric_pack(user_id, pack_id, path=path)  # type: ignore[return-value]


def list_metric_packs(
    user_id: str,
    path: str | None = None,
    workspace_id: str | None = None,
) -> list[MetricPackPublic]:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    with _connect(path) as con:
        if workspace_id:
            rows = con.execute(
                """
                SELECT * FROM metric_packs
                WHERE workspace_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC
                """,
                (workspace_id,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT * FROM metric_packs
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
    return [_pack_from_row(r) for r in rows]


def get_metric_pack(
    user_id: str, pack_id: str, path: str | None = None
) -> MetricPackPublic | None:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT * FROM metric_packs
            WHERE pack_id = ? AND deleted_at IS NULL
            """,
            (pack_id,),
        ).fetchone()
    if not row or not _user_can_access_pack(user_id, row, path=path):
        return None
    return _pack_from_row(row)


def update_metric_pack(
    user_id: str,
    pack_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    config: dict[str, Any] | None = None,
    certified: bool | None = None,
    connection_id: str | None = None,
    clear_connection: bool = False,
    path: str | None = None,
) -> MetricPackPublic | None:
    from config.analysis_config import MetricConfig

    path = path or _auth_db_path()
    existing = get_metric_pack(user_id, pack_id, path=path)
    if not existing:
        return None
    if not _user_can_mutate_pack(user_id, pack_id, path=path):
        return None

    if config is not None:
        MetricConfig(**config)

    if connection_id and not get_connection(user_id, connection_id, path=path):
        raise ValueError("connection_id not found or not accessible")

    now = _utcnow()
    fields: list[str] = []
    params: list[Any] = []

    def _set(col: str, val: Any) -> None:
        fields.append(f"{col} = ?")
        params.append(val)

    if name is not None:
        _set("name", name.strip())
    if description is not None:
        _set("description", description.strip())
    if config is not None:
        _set("config_json", json.dumps(config))
        _set("version", existing.version + 1)
    if certified is not None:
        _set("certified", 1 if certified else 0)
    if clear_connection:
        _set("connection_id", None)
    elif connection_id is not None:
        _set("connection_id", connection_id)

    if not fields:
        return existing

    _set("updated_at", now)
    params.append(pack_id)

    with _connect(path) as con:
        con.execute(
            f"UPDATE metric_packs SET {', '.join(fields)} "
            "WHERE pack_id = ? AND deleted_at IS NULL",
            tuple(params),
        )
    return get_metric_pack(user_id, pack_id, path=path)


def delete_metric_pack(user_id: str, pack_id: str, path: str | None = None) -> bool:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not _user_can_mutate_pack(user_id, pack_id, path=path):
        return False
    now = _utcnow()
    with _connect(path) as con:
        cur = con.execute(
            """
            UPDATE metric_packs SET deleted_at = ?, updated_at = ?
            WHERE pack_id = ? AND deleted_at IS NULL
            """,
            (now, now, pack_id),
        )
        try:
            return int(cur.rowcount or 0) > 0
        except Exception:
            row = con.execute(
                """
                SELECT 1 FROM metric_packs
                WHERE pack_id = ? AND deleted_at IS NOT NULL
                """,
                (pack_id,),
            ).fetchone()
            return row is not None


# ── Schema annotations (Phase 2) ──────────────────────────────────────────────

@dataclass
class SchemaAnnotationsPublic:
    connection_id: str
    annotations: dict[str, Any]
    synonyms: dict[str, str]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "annotations": self.annotations,
            "synonyms": self.synonyms,
            "updated_at": self.updated_at,
        }


def get_annotations(
    user_id: str, connection_id: str, path: str | None = None
) -> SchemaAnnotationsPublic | None:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not get_connection(user_id, connection_id, path=path):
        return None
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT * FROM schema_annotations
            WHERE connection_id = ?
            """,
            (connection_id,),
        ).fetchone()
    if not row:
        return SchemaAnnotationsPublic(
            connection_id=connection_id,
            annotations={},
            synonyms={},
            updated_at="",
        )
    d = dict(row)
    return SchemaAnnotationsPublic(
        connection_id=d["connection_id"],
        annotations=json.loads(d.get("annotations_json") or "{}"),
        synonyms=json.loads(d.get("synonyms_json") or "{}"),
        updated_at=d.get("updated_at") or "",
    )


def upsert_annotations(
    user_id: str,
    connection_id: str,
    *,
    annotations: dict[str, Any],
    synonyms: dict[str, str] | None = None,
    path: str | None = None,
) -> SchemaAnnotationsPublic:
    from agents.analyze.semantic_layer import validate_annotations_payload

    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not get_connection(user_id, connection_id, path=path):
        raise ValueError("connection_id not found or not accessible")
    # Analysts can read; only creator/workspace owner can write annotations
    if not _user_can_mutate_connection(user_id, connection_id, path=path):
        raise PermissionError("Owner role required to edit annotations")

    clean = validate_annotations_payload(annotations)
    syn = synonyms or {}
    if not isinstance(syn, dict):
        raise ValueError("synonyms must be an object")
    clean_syn = {str(k).strip(): str(v).strip() for k, v in syn.items() if str(k).strip()}
    now = _utcnow()

    with _connect(path) as con:
        existing = con.execute(
            "SELECT connection_id FROM schema_annotations WHERE connection_id = ?",
            (connection_id,),
        ).fetchone()
        if existing:
            con.execute(
                """
                UPDATE schema_annotations
                SET annotations_json = ?, synonyms_json = ?, updated_at = ?, user_id = ?
                WHERE connection_id = ?
                """,
                (json.dumps(clean), json.dumps(clean_syn), now, user_id, connection_id),
            )
        else:
            con.execute(
                """
                INSERT INTO schema_annotations
                    (connection_id, user_id, annotations_json, synonyms_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (connection_id, user_id, json.dumps(clean), json.dumps(clean_syn), now),
            )
    return get_annotations(user_id, connection_id, path=path)  # type: ignore[return-value]


def record_schema_snapshot(
    user_id: str,
    connection_id: str,
    *,
    schema_context: str,
    schema_hash: str,
    path: str | None = None,
) -> None:
    """Persist last-seen schema for drift-vs-previous-connect checks."""
    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not get_connection(user_id, connection_id, path=path):
        return
    now = _utcnow()
    # Cap stored schema text
    snap = (schema_context or "")[:50_000]
    with _connect(path) as con:
        con.execute(
            """
            UPDATE db_connections
            SET schema_snapshot_json = ?, schema_hash = ?, schema_snapshot_at = ?, updated_at = ?
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (snap, schema_hash, now, now, connection_id),
        )


def get_schema_snapshot(
    user_id: str, connection_id: str, path: str | None = None
) -> dict[str, Any] | None:
    path = path or _auth_db_path()
    init_workspace_tables(path)
    if not get_connection(user_id, connection_id, path=path):
        return None
    with _connect(path) as con:
        row = con.execute(
            """
            SELECT schema_snapshot_json, schema_hash, schema_snapshot_at
            FROM db_connections
            WHERE connection_id = ? AND deleted_at IS NULL
            """,
            (connection_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    return {
        "schema_context": d.get("schema_snapshot_json") or "",
        "schema_hash": d.get("schema_hash") or "",
        "schema_snapshot_at": d.get("schema_snapshot_at"),
    }
