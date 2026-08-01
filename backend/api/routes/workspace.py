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
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from auth import workspace_store
from config.analysis_config import MetricConfig
from tools.db_tools import DBConnection

from ..deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["workspace"])

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


def _test_pg(
    *,
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str,
    sslmode: str = "prefer",
) -> dict[str, Any]:
    conn = DBConnection(
        backend="postgres",
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
    )
    result = conn.test_connection()
    # Enrich with table names (capped) for onboarding UX
    if result.get("success"):
        try:
            tables = conn.inspect_schema()
            # inspect_schema returns a string — also try listing tables
            from tools.db_tools import DBConnection as _
            names = []
            try:
                names = conn._get_tables_postgres()[:50]
            except Exception:
                pass
            result["tables"] = names
            result["schema_preview"] = (tables or "")[:2000]
        except Exception as exc:
            logger.debug("schema enrich failed: %s", exc)
    return result


# ── Connection models ─────────────────────────────────────────────────────────

class ConnectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    host: str = Field(min_length=1, max_length=253)
    port: int = 5432
    dbname: str = Field(min_length=1, max_length=128)
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=0, max_length=512)
    backend: Literal["postgres"] = "postgres"
    sslmode: str = "prefer"
    test: bool = True  # test before save by default

    @field_validator("sslmode")
    @classmethod
    def _ssl(cls, v: str) -> str:
        if v not in _ALLOWED_SSL:
            raise ValueError(f"sslmode must be one of {sorted(_ALLOWED_SSL)}")
        return v

    @field_validator("port")
    @classmethod
    def _port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port out of range")
        return v


class ConnectionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    host: Optional[str] = Field(default=None, min_length=1, max_length=253)
    port: Optional[int] = None
    dbname: Optional[str] = Field(default=None, min_length=1, max_length=128)
    username: Optional[str] = Field(default=None, min_length=1, max_length=128)
    password: Optional[str] = Field(default=None, max_length=512)
    sslmode: Optional[str] = None

    @field_validator("sslmode")
    @classmethod
    def _ssl(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _ALLOWED_SSL:
            raise ValueError(f"sslmode must be one of {sorted(_ALLOWED_SSL)}")
        return v


class EphemeralTestRequest(BaseModel):
    host: str
    port: int = 5432
    dbname: str
    username: str
    password: str = ""
    sslmode: str = "prefer"


# ── Connection routes ─────────────────────────────────────────────────────────

@router.get("/connections")
async def list_connections(current_user: dict = Depends(get_current_user)):
    items = workspace_store.list_connections(current_user["user_id"])
    return {"connections": [c.to_dict() for c in items]}


@router.post("/connections", status_code=status.HTTP_201_CREATED)
async def create_connection(
    body: ConnectionCreate,
    current_user: dict = Depends(get_current_user),
):
    _validate_host(body.host)
    user_id = current_user["user_id"]

    if body.test:
        result = _test_pg(
            host=body.host, port=body.port, dbname=body.dbname,
            user=body.username, password=body.password, sslmode=body.sslmode,
        )
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=f"Connection test failed: {result.get('error') or 'unknown error'}",
            )

    conn = workspace_store.create_connection(
        user_id,
        name=body.name,
        host=body.host,
        port=body.port,
        dbname=body.dbname,
        username=body.username,
        password=body.password,
        backend=body.backend,
        sslmode=body.sslmode,
    )
    if body.test:
        workspace_store.record_connection_test(user_id, conn.connection_id, ok=True)
        conn = workspace_store.get_connection(user_id, conn.connection_id) or conn

    logger.info("connection.created user=%s id=%s host=%s", user_id, conn.connection_id, body.host)
    return conn.to_dict()


@router.get("/connections/{connection_id}")
async def get_connection(connection_id: str, current_user: dict = Depends(get_current_user)):
    conn = workspace_store.get_connection(current_user["user_id"], connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn.to_dict()


@router.patch("/connections/{connection_id}")
async def update_connection(
    connection_id: str,
    body: ConnectionUpdate,
    current_user: dict = Depends(get_current_user),
):
    if body.host is not None:
        _validate_host(body.host)
    try:
        conn = workspace_store.update_connection(
            current_user["user_id"],
            connection_id,
            name=body.name,
            host=body.host,
            port=body.port,
            dbname=body.dbname,
            username=body.username,
            password=body.password,
            sslmode=body.sslmode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn.to_dict()


@router.delete("/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(connection_id: str, current_user: dict = Depends(get_current_user)):
    ok = workspace_store.delete_connection(current_user["user_id"], connection_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Connection not found")
    return None


@router.post("/connections/{connection_id}/test")
async def test_saved_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    secrets = workspace_store.get_connection_secrets(user_id, connection_id)
    if not secrets:
        raise HTTPException(status_code=404, detail="Connection not found")

    result = _test_pg(
        host=secrets.host, port=secrets.port, dbname=secrets.dbname,
        user=secrets.username, password=secrets.password, sslmode=secrets.sslmode,
    )
    workspace_store.record_connection_test(
        user_id, connection_id,
        ok=bool(result.get("success")),
        error=result.get("error"),
    )
    # Never echo password
    return {
        "success": result.get("success"),
        "error": result.get("error"),
        "table_count": result.get("table_count", 0),
        "tables": result.get("tables", []),
    }


@router.post("/connections/test-ephemeral")
async def test_ephemeral(
    body: EphemeralTestRequest,
    current_user: dict = Depends(get_current_user),
):
    _validate_host(body.host)
    result = _test_pg(
        host=body.host, port=body.port, dbname=body.dbname,
        user=body.username, password=body.password, sslmode=body.sslmode,
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
async def list_packs(current_user: dict = Depends(get_current_user)):
    packs = workspace_store.list_metric_packs(current_user["user_id"])
    return {"metric_packs": [p.to_dict() for p in packs]}


@router.post("/metric-packs", status_code=status.HTTP_201_CREATED)
async def create_pack(
    body: MetricPackCreate,
    current_user: dict = Depends(get_current_user),
):
    try:
        # Validate early for clear 400s
        MetricConfig(**body.config)
        pack = workspace_store.create_metric_pack(
            current_user["user_id"],
            name=body.name,
            description=body.description,
            config=body.config,
            certified=body.certified,
            connection_id=body.connection_id,
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
    pack = workspace_store.get_metric_pack(current_user["user_id"], pack_id)
    if not pack:
        raise HTTPException(status_code=404, detail="Metric pack not found")
    return pack.to_dict()


@router.patch("/metric-packs/{pack_id}")
async def update_pack(
    pack_id: str,
    body: MetricPackUpdate,
    current_user: dict = Depends(get_current_user),
):
    try:
        if body.config is not None:
            MetricConfig(**body.config)
        pack = workspace_store.update_metric_pack(
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
async def delete_pack(pack_id: str, current_user: dict = Depends(get_current_user)):
    ok = workspace_store.delete_metric_pack(current_user["user_id"], pack_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Metric pack not found")
    return None
