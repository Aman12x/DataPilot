"""
backend/api/routes/workspace.py — Saved DB connections + metric packs API.

Endpoints:
  Connections:
    GET    /connections
    POST   /connections
    GET    /connections/{id}
    PATCH  /connections/{id}
    DELETE /connections/{id}
    POST   /connections/{id}/test
    POST   /connections/test-ephemeral   (test without saving)

  Metric packs:
    GET    /metric-packs
    POST   /metric-packs
    GET    /metric-packs/{id}
    PATCH  /metric-packs/{id}
    DELETE /metric-packs/{id}

  Annotations / drift (Phase 2):
    GET/PUT /connections/{id}/annotations
    GET     /connections/{id}/drift?metric_pack_id=
    POST    /connections/{id}/test?metric_pack_id=  (returns drift_warnings)
"""

from __future__ import annotations

import asyncio

import json
import logging
import os
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from auth import org_store, workspace_store
from config.analysis_config import MetricConfig
from tools.db_tools import DBConnection

from ..auth_rate import check_auth_rate
from ..deps import get_current_user, resolve_workspace_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["workspace"])


def _require_owner(user_id: str, workspace_id: str | None) -> None:
    if not workspace_id:
        # Legacy / guest path: user owns their own resources
        return
    try:
        org_store.require_role(user_id, workspace_id, min_role="owner")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

_ALLOWED_SSL = frozenset({"disable", "allow", "prefer", "require", "verify-ca", "verify-full"})
_ALLOW_PRIVATE = os.getenv("ALLOW_PRIVATE_DB_HOSTS", "").lower() in ("1", "true", "yes")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _validate_host(host: str) -> None:
    """Reuse SSRF guard from runs; optionally allow private nets for VPC SMBs."""
    if _ALLOW_PRIVATE:
        if not host or len(host) > 253:
            raise HTTPException(status_code=400, detail="Invalid database host")
        return
    from .runs import _validate_pg_host
    _validate_pg_host(host)


def _validate_bq_credentials(raw: str) -> None:
    if not raw or len(raw) > 65_536:
        raise HTTPException(status_code=400, detail="BigQuery credentials JSON required")
    try:
        info = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="credentials must be valid JSON") from exc
    if not isinstance(info, dict):
        raise HTTPException(status_code=400, detail="credentials must be a JSON object")
    for key in ("type", "client_email", "private_key"):
        if not info.get(key):
            raise HTTPException(
                status_code=400,
                detail=f"Service account JSON missing required field: {key}",
            )


def _build_db_connection(
    *,
    backend: str,
    host: str = "",
    port: int = 0,
    dbname: str = "",
    user: str = "",
    password: str = "",
    sslmode: str = "prefer",
    project_id: str = "",
) -> DBConnection:
    if backend == "bigquery":
        return DBConnection(
            backend="bigquery",
            project_id=project_id,
            dataset=dbname,
            credentials_json=password,
        )
    return DBConnection(
        backend=backend,
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
    )


def _test_db(
    *,
    backend: str,
    host: str = "",
    port: int = 0,
    dbname: str = "",
    user: str = "",
    password: str = "",
    sslmode: str = "prefer",
    project_id: str = "",
) -> dict[str, Any]:
    conn = _build_db_connection(
        backend=backend,
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        project_id=project_id,
    )
    result = conn.test_connection()
    if result.get("success"):
        try:
            preview = conn.inspect_schema()
            names = []
            try:
                names = conn._get_tables()[:50]
            except Exception:
                pass
            result["tables"] = names
            result["schema_preview"] = (preview or "")[:2000]
        except Exception as exc:
            logger.debug("schema enrich failed: %s", exc)
    return result


# Public entrypoint used by routes — tests patch this name for live-test isolation.
def _test_pg(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("backend", "postgres")
    return _test_db(**kwargs)


# Overall ceiling for a connection test. The drivers already carry
# DB_CONNECT_TIMEOUT, but schema inspection runs several queries afterwards, so
# bound the whole thing.
_TEST_TIMEOUT = float(os.getenv("DB_TEST_TIMEOUT", "25"))


async def _test_db_async(**kwargs: Any) -> dict[str, Any]:
    """Run a connection test off the event loop, with a hard ceiling.

    The host is user-supplied, so this must never be able to pin the loop: the
    backend runs one worker, and a blackholed host would otherwise stall every
    other request for as long as the OS TCP timeout.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_test_pg, **kwargs), timeout=_TEST_TIMEOUT
        )
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Connection test timed out after {_TEST_TIMEOUT:.0f}s",
            "table_count": 0,
            "tables": [],
        }


# ── Connection models ─────────────────────────────────────────────────────────

class ConnectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    backend: Literal["postgres", "mysql", "bigquery"] = "postgres"
    host: str = Field(default="", max_length=253)
    port: int = 5432
    dbname: str = Field(default="", max_length=128)  # MySQL/PG db or BQ dataset
    username: str = Field(default="", max_length=128)
    password: str = Field(default="", max_length=65_536)  # BQ service-account JSON
    project_id: str = Field(default="", max_length=128)
    sslmode: str = "prefer"
    test: bool = True  # test before save by default

    @field_validator("sslmode")
    @classmethod
    def _ssl(cls, v: str) -> str:
        if v not in _ALLOWED_SSL:
            raise ValueError(f"sslmode must be one of {sorted(_ALLOWED_SSL)}")
        return v


class ConnectionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    host: Optional[str] = Field(default=None, min_length=1, max_length=253)
    port: Optional[int] = None
    dbname: Optional[str] = Field(default=None, min_length=1, max_length=128)
    username: Optional[str] = Field(default=None, min_length=1, max_length=128)
    password: Optional[str] = Field(default=None, max_length=65_536)
    project_id: Optional[str] = Field(default=None, max_length=128)
    sslmode: Optional[str] = None

    @field_validator("sslmode")
    @classmethod
    def _ssl(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _ALLOWED_SSL:
            raise ValueError(f"sslmode must be one of {sorted(_ALLOWED_SSL)}")
        return v


class EphemeralTestRequest(BaseModel):
    backend: Literal["postgres", "mysql", "bigquery"] = "postgres"
    host: str = ""
    port: int = 5432
    dbname: str = ""
    username: str = ""
    password: str = ""
    project_id: str = ""
    sslmode: str = "prefer"


# ── Connection routes ─────────────────────────────────────────────────────────

@router.get("/connections")
async def list_connections(
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    items = await asyncio.to_thread(
        workspace_store.list_connections,
        current_user["user_id"],
        workspace_id=workspace_id,
    )
    return {"connections": [c.to_dict() for c in items]}


@router.post("/connections", status_code=status.HTTP_201_CREATED)
async def create_connection(
    request: Request,
    body: ConnectionCreate,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    # Creating a connection tests it by default, so it reaches out to a
    # user-chosen host like the explicit test routes do.
    await check_auth_rate(request, bucket="conn_test")
    user_id = current_user["user_id"]
    _require_owner(user_id, workspace_id)

    backend = body.backend
    host = body.host
    port = body.port
    dbname = body.dbname
    username = body.username
    password = body.password
    project_id = body.project_id
    sslmode = body.sslmode

    if backend in ("postgres", "mysql"):
        if not host or not dbname or not username:
            raise HTTPException(status_code=400, detail="host, dbname, and username are required")
        if not (1 <= int(port) <= 65535):
            raise HTTPException(status_code=400, detail="port out of range")
        if backend == "mysql" and port == 5432:
            port = 3306
        _validate_host(host)
    elif backend == "bigquery":
        if not project_id or not dbname:
            raise HTTPException(status_code=400, detail="project_id and dataset (dbname) are required")
        _validate_bq_credentials(password)
        host, port, username, sslmode = "", 0, "", "prefer"

    if body.test:
        result = await _test_db_async(
            backend=backend,
            host=host, port=port, dbname=dbname,
            user=username, password=password, sslmode=sslmode,
            project_id=project_id,
        )
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=f"Connection test failed: {result.get('error') or 'unknown error'}",
            )

    conn = await asyncio.to_thread(
        workspace_store.create_connection,
        user_id,
        name=body.name,
        host=host,
        port=port,
        dbname=dbname,
        username=username,
        password=password,
        backend=backend,
        sslmode=sslmode,
        project_id=project_id,
        workspace_id=workspace_id,
    )
    if body.test:
        await asyncio.to_thread(
            workspace_store.record_connection_test,
            user_id,
            conn.connection_id,
            ok=True,
        )
        conn = await asyncio.to_thread(
            workspace_store.get_connection,
            user_id,
            conn.connection_id,
        ) or conn

    logger.info(
        "connection.created user=%s id=%s backend=%s host=%s project=%s",
        user_id, conn.connection_id, backend, host or "-", project_id or "-",
    )
    return conn.to_dict()


@router.get("/connections/{connection_id}")
async def get_connection(connection_id: str, current_user: dict = Depends(get_current_user)):
    conn = await asyncio.to_thread(
        workspace_store.get_connection,
        current_user["user_id"],
        connection_id,
    )
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn.to_dict()


@router.patch("/connections/{connection_id}")
async def update_connection(
    connection_id: str,
    body: ConnectionUpdate,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    _require_owner(current_user["user_id"], workspace_id)
    if body.host is not None:
        _validate_host(body.host)
    try:
        conn = await asyncio.to_thread(
            workspace_store.update_connection,
            current_user["user_id"],
            connection_id,
            name=body.name,
            host=body.host,
            port=body.port,
            dbname=body.dbname,
            username=body.username,
            password=body.password,
            sslmode=body.sslmode,
            project_id=body.project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn.to_dict()


@router.delete("/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    _require_owner(current_user["user_id"], workspace_id)
    ok = await asyncio.to_thread(
        workspace_store.delete_connection,
        current_user["user_id"],
        connection_id,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Connection not found")
    return None


@router.post("/connections/{connection_id}/test")
async def test_saved_connection(
    request: Request,
    connection_id: str,
    current_user: dict = Depends(get_current_user),
    metric_pack_id: Optional[str] = None,
):
    """
    Live-test a saved connection. Optional `metric_pack_id` query param runs
    pack-vs-live drift checks and returns `drift_warnings`.
    """
    await check_auth_rate(request, bucket="conn_test")
    user_id = current_user["user_id"]
    secrets = await asyncio.to_thread(
        workspace_store.get_connection_secrets,
        user_id,
        connection_id,
    )
    if not secrets:
        raise HTTPException(status_code=404, detail="Connection not found")

    result = await _test_db_async(
        backend=secrets.backend,
        host=secrets.host, port=secrets.port, dbname=secrets.dbname,
        user=secrets.username, password=secrets.password, sslmode=secrets.sslmode,
        project_id=secrets.project_id,
    )
    await asyncio.to_thread(
        workspace_store.record_connection_test,
        user_id,
        connection_id,
        ok=bool(result.get("success")),
        error=result.get("error"),
    )

    drift_warnings: list[str] = []
    schema_changed = False
    if result.get("success"):
        try:
            from agents.analyze.semantic_layer import (
                detect_pack_drift,
                schema_hash,
                strip_sql_dialect_header,
            )
            ann = await asyncio.to_thread(
                workspace_store.get_annotations,
                user_id,
                connection_id,
            )
            annotations = (ann.annotations if ann else {}) or {}
            conn = _build_db_connection(
                backend=secrets.backend,
                host=secrets.host, port=secrets.port, dbname=secrets.dbname,
                user=secrets.username, password=secrets.password, sslmode=secrets.sslmode,
                project_id=secrets.project_id,
            )
            schema_context = conn.inspect_schema(annotations=annotations)
            sch_hash = schema_hash(schema_context)
            prev = await asyncio.to_thread(
                workspace_store.get_schema_snapshot,
                user_id,
                connection_id,
            )
            if prev and prev.get("schema_hash") and prev["schema_hash"] != sch_hash:
                schema_changed = True
                drift_warnings.append(
                    "Live schema hash differs from last successful connect — "
                    "columns or tables may have changed."
                )
            await asyncio.to_thread(
                workspace_store.record_schema_snapshot,
                user_id,
                connection_id,
                schema_context=strip_sql_dialect_header(schema_context),
                schema_hash=sch_hash,
            )
            if metric_pack_id:
                pack = await asyncio.to_thread(
                    workspace_store.get_metric_pack,
                    user_id,
                    metric_pack_id,
                )
                if pack:
                    drift_warnings.extend(
                        detect_pack_drift(MetricConfig(**pack.config), schema_context)
                    )
        except Exception as exc:
            logger.debug("connection test drift enrich failed: %s", exc)

    # Never echo password
    return {
        "success": result.get("success"),
        "error": result.get("error"),
        "table_count": result.get("table_count", 0),
        "tables": result.get("tables", []),
        "schema_changed": schema_changed,
        "drift_warnings": drift_warnings,
    }


@router.post("/connections/test-ephemeral")
async def test_ephemeral(
    request: Request,
    body: EphemeralTestRequest,
    current_user: dict = Depends(get_current_user),
):
    await check_auth_rate(request, bucket="conn_test")
    if body.backend in ("postgres", "mysql"):
        _validate_host(body.host)
    elif body.backend == "bigquery":
        _validate_bq_credentials(body.password)
        if not body.project_id or not body.dbname:
            raise HTTPException(status_code=400, detail="project_id and dataset required")
    result = await _test_db_async(
        backend=body.backend,
        host=body.host, port=body.port, dbname=body.dbname,
        user=body.username, password=body.password, sslmode=body.sslmode,
        project_id=body.project_id,
    )
    return {
        "success": result.get("success"),
        "error": result.get("error"),
        "table_count": result.get("table_count", 0),
        "tables": result.get("tables", []),
    }


# ── Metric pack models ────────────────────────────────────────────────────────

class MetricPackCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=2000)
    config: dict[str, Any]
    certified: bool = False
    connection_id: Optional[str] = None


class MetricPackUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    config: Optional[dict[str, Any]] = None
    certified: Optional[bool] = None
    connection_id: Optional[str] = None
    clear_connection: bool = False


# ── Metric pack routes ────────────────────────────────────────────────────────

@router.get("/metric-packs")
async def list_packs(
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    packs = await asyncio.to_thread(
        workspace_store.list_metric_packs,
        current_user["user_id"],
        workspace_id=workspace_id,
    )
    return {"metric_packs": [p.to_dict() for p in packs]}


@router.post("/metric-packs", status_code=status.HTTP_201_CREATED)
async def create_pack(
    body: MetricPackCreate,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    _require_owner(current_user["user_id"], workspace_id)
    try:
        # Validate early for clear 400s
        MetricConfig(**body.config)
        pack = await asyncio.to_thread(
            workspace_store.create_metric_pack,
            current_user["user_id"],
            name=body.name,
            description=body.description,
            config=body.config,
            certified=body.certified,
            connection_id=body.connection_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Pydantic validation errors
        raise HTTPException(status_code=400, detail=f"Invalid metric config: {exc}") from exc

    logger.info("metric_pack.created user=%s id=%s certified=%s",
                current_user["user_id"], pack.pack_id, body.certified)
    return pack.to_dict()


@router.get("/metric-packs/{pack_id}")
async def get_pack(pack_id: str, current_user: dict = Depends(get_current_user)):
    pack = await asyncio.to_thread(
        workspace_store.get_metric_pack,
        current_user["user_id"],
        pack_id,
    )
    if not pack:
        raise HTTPException(status_code=404, detail="Metric pack not found")
    return pack.to_dict()


@router.patch("/metric-packs/{pack_id}")
async def update_pack(
    pack_id: str,
    body: MetricPackUpdate,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    _require_owner(current_user["user_id"], workspace_id)
    try:
        if body.config is not None:
            MetricConfig(**body.config)
        pack = await asyncio.to_thread(
            workspace_store.update_metric_pack,
            current_user["user_id"],
            pack_id,
            name=body.name,
            description=body.description,
            config=body.config,
            certified=body.certified,
            connection_id=body.connection_id,
            clear_connection=body.clear_connection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metric config: {exc}") from exc
    if not pack:
        raise HTTPException(status_code=404, detail="Metric pack not found")
    return pack.to_dict()


@router.delete("/metric-packs/{pack_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pack(
    pack_id: str,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    _require_owner(current_user["user_id"], workspace_id)
    ok = await asyncio.to_thread(
        workspace_store.delete_metric_pack,
        current_user["user_id"],
        pack_id,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Metric pack not found")
    return None


# ── Schema annotations (Phase 2) ──────────────────────────────────────────────

class AnnotationsUpsert(BaseModel):
    annotations: dict[str, Any]
    synonyms: dict[str, str] = Field(default_factory=dict)


@router.get("/connections/{connection_id}/annotations")
async def get_annotations(connection_id: str, current_user: dict = Depends(get_current_user)):
    ann = await asyncio.to_thread(
        workspace_store.get_annotations,
        current_user["user_id"],
        connection_id,
    )
    if ann is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return ann.to_dict()


@router.put("/connections/{connection_id}/annotations")
async def put_annotations(
    connection_id: str,
    body: AnnotationsUpsert,
    current_user: dict = Depends(get_current_user),
):
    try:
        ann = await asyncio.to_thread(
            workspace_store.upsert_annotations,
            current_user["user_id"],
            connection_id,
            annotations=body.annotations,
            synonyms=body.synonyms,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ann.to_dict()


@router.get("/connections/{connection_id}/drift")
async def connection_drift(
    connection_id: str,
    metric_pack_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Compare last-known schema snapshot / live connect against a metric pack.
    Uses the stored snapshot when available (no live DB round-trip required).
    """
    user_id = current_user["user_id"]
    snap = await asyncio.to_thread(workspace_store.get_schema_snapshot, user_id, connection_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    warnings: list[str] = []
    if not snap.get("schema_context"):
        return {
            "connection_id": connection_id,
            "schema_hash": "",
            "has_snapshot": False,
            "drift_warnings": ["No schema snapshot yet — run a connection test first."],
        }
    if metric_pack_id:
        pack = await asyncio.to_thread(workspace_store.get_metric_pack, user_id, metric_pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="Metric pack not found")
        from agents.analyze.semantic_layer import detect_pack_drift
        warnings = detect_pack_drift(MetricConfig(**pack.config), snap["schema_context"])
    return {
        "connection_id": connection_id,
        "schema_hash": snap.get("schema_hash") or "",
        "has_snapshot": True,
        "schema_snapshot_at": snap.get("schema_snapshot_at"),
        "drift_warnings": warnings,
    }


# ── Verified queries (item 6: "teach it how we write queries") ────────────────

class VerifiedQueryCreate(BaseModel):
    task: str = Field(min_length=3, max_length=2000)
    sql: str = Field(min_length=5, max_length=20000)
    name: str = Field(default="", max_length=120)
    connection_id: str = ""


@router.get("/verified-queries")
async def list_verified_queries_endpoint(
    connection_id: str | None = None,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    from memory.verified_queries import list_verified_queries

    rows = await asyncio.to_thread(
        list_verified_queries,
        current_user["user_id"],
        workspace_id or "",
        connection_id,
    )
    return {"verified_queries": rows}


@router.post("/verified-queries", status_code=status.HTTP_201_CREATED)
async def create_verified_query(
    body: VerifiedQueryCreate,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    user_id = current_user["user_id"]
    if user_id.startswith("guest-"):
        # Guest identities are disposable; their exemplars would be orphans.
        raise HTTPException(status_code=403, detail="Sign up to save verified queries.")
    _require_owner(user_id, workspace_id)

    from tools.db_tools import validate_sql
    try:
        validate_sql(body.sql)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Rejected SQL: {exc}") from exc

    from memory.verified_queries import add_verified_query
    try:
        vq_id = await asyncio.to_thread(
            add_verified_query,
            body.task,
            body.sql,
            source="contributed",
            user_id=user_id,
            workspace_id=workspace_id or "",
            name=body.name,
            connection_id=body.connection_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("verified_query.created user=%s id=%s", user_id, vq_id)
    return {"vq_id": vq_id}


@router.delete("/verified-queries/{vq_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_verified_query_endpoint(
    vq_id: str,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    _require_owner(current_user["user_id"], workspace_id)
    from memory.verified_queries import delete_verified_query

    removed = await asyncio.to_thread(
        delete_verified_query,
        vq_id,
        current_user["user_id"],
        workspace_id or "",
    )
    if not removed:
        raise HTTPException(status_code=404, detail="Verified query not found")
