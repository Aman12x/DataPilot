"""
backend/api/routes/runs.py

POST  /runs                  {task, db_backend?}   → {run_id}
GET   /runs/{id}/stream-token                      → {stream_token}
GET   /runs/{id}/stream      ?stream_token=...      → SSE stream
POST  /runs/{id}/resume      {gate, value}         → {status: "ok"}
GET   /runs                  ?limit=10             → list of past runs
GET   /runs/{id}/detail                            → run detail (narrative, recommendation)
GET   /runs/{id}/pdf-token                         → {pdf_token}
GET   /runs/{id}/pdf         ?pdf_token=...         → PDF bytes
GET   /health                                      → {status: "ok"}
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from agents.analyze.prompt_safety import strip_delimiters

from ..auth_rate import client_ip
from ..budget import check_budget, scope_for
from ..deps import (
    create_pdf_token,
    create_stream_token,
    get_current_user,
    resolve_workspace_id,
    verify_scoped_token,
)
from ..run_manager import (
    check_rate_limit,
    check_resume_rate_limit,
    cleanup_run,
    get_cached_error,
    get_gate_deadline,
    get_owner,
    read_result,
    resume_run,
    set_gate_deadline,
    start_run,
)
from .upload import resolve_upload_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["runs"])

# ── Input sanitisation ────────────────────────────────────────────────────────

_MAX_TASK_LEN = 1000

_PRIVATE_NETS = [
    ipaddress.ip_network(n) for n in (
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "127.0.0.0/8", "169.254.0.0/16", "::1/128", "fc00::/7",
    )
]

_INJECT_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions?|you\s+are\s+now|system\s*:)",
    re.IGNORECASE,
)


def _sanitise_task(task: str) -> str:
    task = task.strip()
    if not task:
        raise HTTPException(status_code=422, detail="Task must not be empty")
    if len(task) > _MAX_TASK_LEN:
        raise HTTPException(status_code=422, detail=f"Task too long (max {_MAX_TASK_LEN} chars)")
    if _INJECT_RE.search(task):
        raise HTTPException(status_code=422, detail="Task contains disallowed content")
    return task


# Resume payloads carry analyst free text (gate notes, recommendation
# rewrites, edited SQL). They are persisted to run history and replayed into
# later prompts, so they get the same treatment as the initial task. The cap is
# larger because an edited SQL statement legitimately runs long.
_MAX_RESUME_STR = 10_000
_MAX_RESUME_FIELDS = 50
_MAX_RESUME_DEPTH = 4


def _sanitise_resume_value(value: Any, _depth: int = 0) -> Any:
    """Recursively bound and screen a gate resume payload.

    Mirrors _sanitise_task: strip the delimiter marker so stored text cannot
    close a wrapper it is later embedded in, cap length, and reject the same
    override phrases.
    """
    if _depth > _MAX_RESUME_DEPTH:
        raise HTTPException(status_code=422, detail="Resume payload nested too deeply")

    if isinstance(value, str):
        cleaned = strip_delimiters(value)
        if len(cleaned) > _MAX_RESUME_STR:
            raise HTTPException(
                status_code=422,
                detail=f"Resume field too long (max {_MAX_RESUME_STR} chars)",
            )
        if _INJECT_RE.search(cleaned):
            raise HTTPException(status_code=422, detail="Resume payload contains disallowed content")
        return cleaned

    if isinstance(value, dict):
        if len(value) > _MAX_RESUME_FIELDS:
            raise HTTPException(status_code=422, detail="Resume payload has too many fields")
        return {k: _sanitise_resume_value(v, _depth + 1) for k, v in value.items()}

    if isinstance(value, list):
        if len(value) > _MAX_RESUME_FIELDS:
            raise HTTPException(status_code=422, detail="Resume payload has too many entries")
        return [_sanitise_resume_value(v, _depth + 1) for v in value]

    return value


def _validate_pg_host(host: str) -> None:
    if not host:
        return
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        if any(addr in net for net in _PRIVATE_NETS):
            raise HTTPException(status_code=400, detail=f"Database host '{host}' is not allowed")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Could not resolve database host '{host}'",
        )


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_ALLOWED_GATES = frozenset({
    "semantic_cache", "intent", "metric", "query", "analysis", "narrative", "srm",
})

_ALLOWED_DB_BACKENDS = frozenset({"duckdb", "postgres", "mysql", "bigquery"})


# ── JSON helpers ──────────────────────────────────────────────────────────────

class _JsonEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _get_graph(request: Request) -> Any:
    return request.app.state.graph


def _get_memory_store(request: Request) -> Any:
    return request.app.state.memory_store


def _user_from_stream_token(stream_token: str, run_id: str) -> dict[str, str]:
    return verify_scoped_token(stream_token, "stream", run_id)


def _workspace_of_run(graph: Any, run_id: str) -> tuple[str | None, str | None]:
    """Return (owner_user_id, workspace_id) from graph state or memory store.

    Two synchronous SQLite reads. Blocking — call via `asyncio.to_thread`.
    """
    state_uid: str | None = None
    ws_id: str | None = None
    config = {"configurable": {"thread_id": run_id}}
    try:
        state = graph.get_state(config)
        values = state.values if hasattr(state, "values") else {}
        if values:
            state_uid = values.get("user_id") or None
            ws_id = values.get("workspace_id") or None
    except Exception:
        pass
    if not ws_id or not state_uid:
        try:
            from memory.store import get_run
            row = get_run(run_id)
            if row:
                state_uid = state_uid or row.get("user_id") or None
                ws_id = ws_id or row.get("workspace_id") or None
        except Exception:
            pass
    return state_uid, ws_id


async def _check_run_access(
    graph: Any,
    run_id: str,
    user_id: str,
    *,
    mutate: bool = False,
) -> None:
    """
    Authorise access to a run.

    - mutate=True  → creator only (resume, live stream tokens)
    - mutate=False → creator OR workspace teammate (history detail/PDF)
    """
    owner = await get_owner(run_id)
    if owner == user_id:
        return

    state_uid, ws_id = await asyncio.to_thread(_workspace_of_run, graph, run_id)
    if state_uid == user_id:
        return

    if not mutate and ws_id:
        from auth.org_store import get_membership
        if await asyncio.to_thread(get_membership, user_id, ws_id) is not None:
            return

    if owner is not None or state_uid:
        raise HTTPException(status_code=403, detail="Not your run")
    raise HTTPException(status_code=403, detail="Not your run")


async def _check_ownership(graph: Any, run_id: str, user_id: str) -> None:
    """Creator-only check (live gates / resume)."""
    await _check_run_access(graph, run_id, user_id, mutate=True)


async def _check_parent_ownership(graph: Any, parent_run_id: str, user_id: str) -> None:
    if not _UUID_RE.match(parent_run_id):
        raise HTTPException(status_code=400, detail="Invalid parent_run_id")
    # Follow-ups allowed for workspace teammates (read + branch)
    await _check_run_access(graph, parent_run_id, user_id, mutate=False)


def _snap_to_interrupt_payload(graph: Any, run_id: str) -> dict | None:
    """Blocking: reads the checkpoint. Call via `asyncio.to_thread`."""
    config = {"configurable": {"thread_id": run_id}}
    try:
        state = graph.get_state(config)
        for task in (state.tasks or []):
            if hasattr(task, "interrupts"):
                for interrupt in task.interrupts:
                    return interrupt.value
    except Exception:
        pass
    return None


# ── Request models ────────────────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    task:          str
    analysis_mode: str = ""       # empty = auto-detect via resolve_task_intent
    db_backend:    str = "duckdb"
    duckdb_path:   str = ""
    connection_id: str = ""       # saved DB connection (preferred over inline pg_*)
    metric_pack_id: str = ""      # saved metric pack
    pg_host:       str = ""
    pg_port:       int = 5432
    pg_dbname:     str = ""
    pg_user:       str = ""
    pg_password:   str = ""
    pg_sslmode:    str = "prefer"
    # BigQuery (inline ephemeral; prefer connection_id in production)
    bq_project_id: str = ""
    bq_dataset: str = ""
    bq_credentials_json: str = ""
    parent_run_id: str = ""       # set for follow-up queries; injects parent narrative as context

    @field_validator("analysis_mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("", "ab_test", "general"):
            raise ValueError("analysis_mode must be 'ab_test', 'general', or '' (auto)")
        return v

    @field_validator("db_backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        if v not in _ALLOWED_DB_BACKENDS:
            raise ValueError(
                "db_backend must be one of: duckdb, postgres, mysql, bigquery"
            )
        return v

    @field_validator("pg_port")
    @classmethod
    def _check_port(cls, v: int) -> int:
        if v == 0:
            return v
        if not (1 <= v <= 65535):
            raise ValueError("pg_port out of range")
        return v

    @field_validator("connection_id", "metric_pack_id", "parent_run_id")
    @classmethod
    def _check_uuidish(cls, v: str) -> str:
        if v and not _UUID_RE.match(v):
            raise ValueError("must be a valid UUID")
        return v


class ResumeRequest(BaseModel):
    gate:  str = Field(min_length=1, max_length=64)
    value: dict = Field(default_factory=dict)

    @field_validator("gate")
    @classmethod
    def _check_gate(cls, v: str) -> str:
        if v not in _ALLOWED_GATES:
            raise ValueError(f"Unknown gate: {v}")
        return v


# ── Routes ────────────────────────────────────────────────────────────────────

_GATE_TIMEOUT_SECS = int(os.getenv("GATE_TIMEOUT_SECONDS", str(4 * 60 * 60)))  # 4 h default


@router.get("/health")
async def health(request: Request):
    """
    Real dependency check — used by Railway's healthcheck path.
    Returns 200 only when all critical systems are reachable.
    """
    checks: dict[str, str] = {}

    # Graph
    try:
        graph = _get_graph(request)
        checks["graph"] = "ok" if graph else "not_initialized"
    except Exception as exc:
        checks["graph"] = f"error: {exc}"

    # SQLite memory DB
    try:
        import sqlite3, os as _os
        db = _os.getenv("MEMORY_DB_PATH", "memory/datapilot_memory.db")
        if _os.path.exists(db):
            sqlite3.connect(db).execute("SELECT 1").fetchone()
            checks["memory_db"] = "ok"
        else:
            checks["memory_db"] = "not_created_yet"
    except Exception as exc:
        checks["memory_db"] = f"error: {exc}"

    # Checkpointer backend — surfaces the split-brain fallback (DATABASE_URL
    # set but checkpoints on local SQLite) instead of hiding it in boot logs.
    checks["checkpointer"] = getattr(request.app.state, "checkpoint_backend", "unknown")

    # Redis (optional)
    from ..run_manager import get_redis_client
    redis = get_redis_client()
    if redis:
        try:
            await redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"
    else:
        checks["redis"] = "not_configured"

    failed = [k for k, v in checks.items() if v.startswith("error")]
    status_code = 503 if failed else 200
    return Response(
        content=json.dumps({"status": "ok" if not failed else "degraded", "checks": checks}),
        media_type="application/json",
        status_code=status_code,
    )


@router.post("/runs", status_code=status.HTTP_201_CREATED)
async def create_run(
    req: StartRunRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    t0 = time.perf_counter()
    user_id = current_user["user_id"]

    ip = client_ip(request)
    budget_scope = scope_for(user_id, ip)
    # Rate limit and budget share the scope: keying either on user_id would let
    # a guest reset it by minting a fresh identity via POST /auth/guest.
    await check_rate_limit(budget_scope)
    await check_budget(user_id, ip)

    task = _sanitise_task(req.task)

    # ── Resolve saved connection (preferred over inline pg_*) ─────────────────
    from auth import workspace_store
    from config.analysis_config import MetricConfig

    connection_id = req.connection_id
    pg_host = req.pg_host
    pg_port = req.pg_port
    pg_dbname = req.pg_dbname
    pg_user = req.pg_user
    pg_password = req.pg_password
    pg_sslmode = req.pg_sslmode
    db_backend = req.db_backend
    bq_project_id = req.bq_project_id
    bq_dataset = req.bq_dataset
    bq_credentials_json = req.bq_credentials_json

    if connection_id:
        secrets = await asyncio.to_thread(
            workspace_store.get_connection_secrets,
            user_id,
            connection_id,
        )
        if not secrets:
            raise HTTPException(status_code=404, detail="Connection not found")
        db_backend = secrets.backend
        if secrets.backend == "bigquery":
            bq_project_id = secrets.project_id
            bq_dataset = secrets.dbname
            bq_credentials_json = secrets.password
            pg_host = pg_user = pg_password = ""
            pg_dbname = ""
            pg_port = 0
        else:
            pg_host = secrets.host
            pg_port = secrets.port
            pg_dbname = secrets.dbname
            pg_user = secrets.username
            pg_password = secrets.password
            pg_sslmode = secrets.sslmode
    elif db_backend == "bigquery":
        if not (bq_project_id and bq_dataset and bq_credentials_json):
            raise HTTPException(
                status_code=400,
                detail="BigQuery requires project_id, dataset, and credentials JSON "
                       "(or a saved connection_id)",
            )
    elif pg_host:
        _validate_pg_host(pg_host)

    # ── Resolve metric pack ───────────────────────────────────────────────────
    metric_config = None
    metric_pack_certified = False
    metric_pack_id = req.metric_pack_id
    metric_pack_version = None
    if metric_pack_id:
        pack = await asyncio.to_thread(workspace_store.get_metric_pack, user_id, metric_pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="Metric pack not found")
        try:
            metric_config = MetricConfig(**pack.config)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Metric pack config invalid: {exc}"
            ) from exc
        metric_pack_certified = bool(pack.certified)
        metric_pack_version = int(pack.version)

    graph  = _get_graph(request)
    run_id = str(uuid.uuid4())

    resolved_duckdb_path = ""
    if req.duckdb_path:
        resolved_duckdb_path = resolve_upload_path(req.duckdb_path, user_id)

    # Extract parent narrative for follow-up context injection
    context_narrative = ""
    if req.parent_run_id:
        await _check_parent_ownership(graph, req.parent_run_id, user_id)
        try:
            parent_config = {"configurable": {"thread_id": req.parent_run_id}}
            parent_state  = await asyncio.to_thread(graph.get_state, parent_config)
            parent_values = parent_state.values if hasattr(parent_state, "values") else {}
            raw_narrative = (
                parent_values.get("final_narrative")
                or parent_values.get("narrative_draft", "")
            )
            context_narrative = raw_narrative[:2000] if raw_narrative else ""
        except Exception:
            logger.warning("Could not read parent run state for %s", req.parent_run_id)

    initial_state: dict[str, Any] = {
        "task":                   task,
        "analysis_mode":          req.analysis_mode,
        "db_backend":             db_backend,
        "duckdb_path":            resolved_duckdb_path,
        "connection_id":          connection_id,
        "pg_host":                pg_host,
        "pg_port":                pg_port,
        "pg_dbname":              pg_dbname,
        "pg_user":                pg_user,
        "pg_password":            pg_password,
        "pg_sslmode":             pg_sslmode,
        "bq_project_id":          bq_project_id,
        "bq_dataset":             bq_dataset,
        "bq_credentials_json":    bq_credentials_json,
        "metric_pack_id":         metric_pack_id,
        "metric_pack_version":    metric_pack_version,
        "metric_pack_certified":  metric_pack_certified,
        "user_id":                user_id,
        "workspace_id":           workspace_id,
        "run_id":                 run_id,
        "context_narrative":      context_narrative,
    }
    if metric_config is not None:
        initial_state["metric_config"] = metric_config
        initial_state["metric"] = metric_config.primary_metric
        initial_state["covariate"] = metric_config.covariate

    await start_run(
        graph,
        run_id,
        initial_state,
        user_id=user_id,
        budget_scope=budget_scope,
    )

    logger.info("run.start user=%s run=%s mode=%s backend=%s latency_ms=%.0f",
                current_user["user_id"], run_id, req.analysis_mode, req.db_backend,
                (time.perf_counter() - t0) * 1000)
    return {"run_id": run_id}


@router.get("/runs/{run_id}/stream-token")
async def stream_token(
    run_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    graph = _get_graph(request)
    await _check_ownership(graph, run_id, current_user["user_id"])
    token = create_stream_token(current_user["user_id"], run_id)
    return {"stream_token": token}


@router.get("/runs/{run_id}/stream")
async def stream_run(
    run_id: str,
    request: Request,
    stream_token: str = Query(...),
    last_id: str = Query(default="$"),  # pass Last-Event-ID on reconnect
):
    current_user = _user_from_stream_token(stream_token, run_id)
    graph        = _get_graph(request)
    await _check_ownership(graph, run_id, current_user["user_id"])

    effective_last_id = last_id

    async def event_generator():
        nonlocal effective_last_id

        # Fast path for reconnects after a crash — no 30s hang
        cached_err = await get_cached_error(run_id)
        if cached_err:
            yield {"data": json.dumps({"type": "error", "message": cached_err})}
            return

        while True:
            if await request.is_disconnected():
                break

            # On reconnect: if a gate interrupt is already pending in the graph
            # state, replay it immediately without blocking on the queue.
            # This handles the case where the graph hit an interrupt before the
            # SSE client connected (e.g. intent gate fires during fast startup).
            interrupt_payload = await asyncio.to_thread(
                _snap_to_interrupt_payload,
                graph,
                run_id,
            )
            if interrupt_payload is not None:
                gate    = interrupt_payload.get("gate", "unknown")
                expires = int(time.time()) + _GATE_TIMEOUT_SECS
                logger.info("run.gate (replay) run=%s gate=%s", run_id, gate)
                await set_gate_deadline(run_id, expires)
                yield {
                    "data": json.dumps({
                        "type":       "gate",
                        "gate":       gate,
                        "payload":    interrupt_payload,
                        "expires_at": expires,
                    }, cls=_JsonEncoder),
                    "id": effective_last_id,
                }
                return

            item = await read_result(run_id, effective_last_id)

            if item is None:
                yield {"comment": "keepalive"}
                continue

            if "_stream_id" in item:
                effective_last_id = item["_stream_id"]

            # Forward Chain-of-Thought step events directly
            if item.get("type") == "step":
                yield {"data": json.dumps(item)}
                continue

            if not item.get("ok"):
                cleanup_run(run_id)
                logger.error("run.error run=%s: %s", run_id, item.get("error"))
                yield {"data": json.dumps({"type": "error", "message": item.get("error", "Unknown error")})}
                return

            interrupt_payload = await asyncio.to_thread(
                _snap_to_interrupt_payload,
                graph,
                run_id,
            )

            if interrupt_payload is not None:
                gate    = interrupt_payload.get("gate", "unknown")
                expires = int(time.time()) + _GATE_TIMEOUT_SECS
                logger.info("run.gate run=%s gate=%s expires=%s", run_id, gate, expires)
                await set_gate_deadline(run_id, expires)

                yield {
                    "data": json.dumps({
                        "type":            "gate",
                        "gate":            gate,
                        "payload":         interrupt_payload,
                        "expires_at":      expires,   # unix timestamp — frontend can show countdown
                    }, cls=_JsonEncoder),
                    "id": effective_last_id,
                }
                return  # EventSource auto-reconnects when user resumes

            # Terminal
            config = {"configurable": {"thread_id": run_id}}
            try:
                final_state = await asyncio.to_thread(graph.get_state, config)
                # Guard against stale "ok" from an intermediate invoke (race condition):
                # if the graph still has pending nodes, this invoke ended at a gate that
                # is about to interrupt — keep waiting for the actual terminal invoke.
                if final_state.next:
                    continue
                state_values = final_state.values if hasattr(final_state, "values") else {}
            except Exception:
                state_values = item.get("snap") or {}

            cleanup_run(run_id)
            logger.info("run.done run=%s user=%s", run_id, current_user["user_id"])
            yield {
                "data": json.dumps({
                    "type":  "done",
                    "state": {
                        "narrative_draft":  state_values.get("final_narrative") or state_values.get("narrative_draft", ""),
                        "recommendation":   state_values.get("recommendation", ""),
                        "run_id":           run_id,
                        "charts":           state_values.get("charts", []),
                        "trust_indicators": state_values.get("trust_indicators", {}),
                        "analysis_mode":    state_values.get("analysis_mode", ""),
                        "deck_data":        state_values.get("deck_data") or {},
                    },
                }, cls=_JsonEncoder)
            }
            return

    return EventSourceResponse(event_generator())


@router.post("/runs/{run_id}/resume")
async def resume_run_endpoint(
    run_id: str,
    req:    ResumeRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    graph = _get_graph(request)
    await _check_ownership(graph, run_id, current_user["user_id"])

    # A resume restarts the graph and spends more tokens, so it faces the same
    # budget check as a fresh run — plus its own, looser rate bucket (a normal
    # run answers ~5 gates in minutes, so the run limit would break it).
    ip = client_ip(request)
    await check_resume_rate_limit(scope_for(current_user["user_id"], ip))
    await check_budget(current_user["user_id"], ip)

    # Reject resume if the gate window has expired
    deadline = await get_gate_deadline(run_id)
    if deadline is not None and time.time() > deadline:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Gate expired — please start a new analysis",
        )

    logger.info("run.resume run=%s gate=%s user=%s", run_id, req.gate, current_user["user_id"])
    await resume_run(graph, run_id, _sanitise_resume_value(req.value))
    return {"status": "ok"}


@router.get("/runs")
def list_runs(
    request: Request,
    limit: int = Query(default=10, le=100),
    current_user: dict = Depends(get_current_user),
    workspace_id: str | None = Depends(resolve_workspace_id),
):
    store = _get_memory_store(request)
    try:
        if workspace_id:
            runs = store.get_all_runs(workspace_id=workspace_id, limit=limit)
        else:
            runs = store.get_all_runs(user_id=current_user["user_id"], limit=limit)
    except Exception as exc:
        logger.warning("list_runs failed: %s", exc)
        return []

    # Enrich with username for team history UI (best-effort)
    try:
        from auth.store import get_user_by_id
        for r in runs:
            if not isinstance(r, dict):
                continue
            uid = r.get("user_id") or ""
            if uid and not r.get("username"):
                user = get_user_by_id(uid)
                r["username"] = user.username if user else ""
    except Exception:
        pass
    return runs


@router.get("/runs/{run_id}/detail")
async def get_run_detail(
    run_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    graph = _get_graph(request)
    await _check_run_access(graph, run_id, current_user["user_id"], mutate=False)
    config = {"configurable": {"thread_id": run_id}}
    try:
        state  = await asyncio.to_thread(graph.get_state, config)
        values = state.values if hasattr(state, "values") else {}
    except Exception:
        raise HTTPException(status_code=404, detail="Run state not found")
    return {
        "run_id":         run_id,
        "task":           values.get("task", ""),
        "narrative":      values.get("final_narrative") or values.get("narrative_draft", ""),
        "recommendation": values.get("recommendation", ""),
        "user_id":        values.get("user_id", ""),
        "workspace_id":   values.get("workspace_id") or "",
    }


@router.get("/runs/{run_id}/pdf-token")
async def pdf_token(
    run_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    graph = _get_graph(request)
    await _check_run_access(graph, run_id, current_user["user_id"], mutate=False)
    return {"pdf_token": create_pdf_token(current_user["user_id"], run_id)}


@router.get("/runs/{run_id}/pdf")
async def get_pdf(
    run_id: str,
    request: Request,
    pdf_token: str = Query(...),
):
    current_user = verify_scoped_token(pdf_token, "pdf", run_id)
    graph        = _get_graph(request)
    await _check_run_access(graph, run_id, current_user["user_id"], mutate=False)

    config = {"configurable": {"thread_id": run_id}}
    try:
        state  = await asyncio.to_thread(graph.get_state, config)
        values = state.values if hasattr(state, "values") else {}
    except Exception:
        raise HTTPException(status_code=404, detail="Run state not found")

    task           = values.get("task", "")
    narrative      = values.get("final_narrative") or values.get("narrative_draft", "")
    recommendation = values.get("recommendation", "")

    try:
        from ..pdf import build_pdf
        # Rendering is CPU-bound reportlab work, not I/O, but it still cannot
        # run here: with --workers 1 it stalls every other request for as long
        # as the document takes.
        pdf_bytes = await asyncio.to_thread(
            build_pdf,
            task=task,
            narrative=narrative,
            recommendation=recommendation,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="datapilot-{run_id[:8]}.pdf"'},
        )
    except Exception as exc:
        logger.exception("PDF generation failed for run %s", run_id)
        raise HTTPException(status_code=500, detail="PDF generation failed")
