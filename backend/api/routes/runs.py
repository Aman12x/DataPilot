"""
backend/api/routes/runs.py

POST  /runs                  {task, db_backend?}   → {run_id}
GET   /runs/{id}/stream      ?token=...&last_id=   → SSE stream
POST  /runs/{id}/resume      {gate, value}         → {status: "ok"}
GET   /runs                  ?limit=10             → list of past runs
GET   /runs/{id}/detail                            → run detail (narrative, recommendation)
GET   /runs/{id}/pdf         ?token=...            → PDF bytes
GET   /health                                      → {status: "ok"}
"""
from __future__ import annotations

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
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

from ..deps import ALGORITHM, SECRET_KEY, get_current_user
from ..run_manager import (
    check_rate_limit,
    cleanup_run,
    get_cached_error,
    get_owner,
    read_result,
    resume_run,
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
        pass  # DNS failure or hostname — let it fail at connect time


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


def _user_from_token_param(token: str) -> dict[str, str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Wrong token type")
    return {"user_id": payload["sub"], "username": payload.get("username", "")}


async def _check_ownership(graph: Any, run_id: str, user_id: str) -> None:
    owner = await get_owner(run_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="Not your run")
    if owner is None:
        config = {"configurable": {"thread_id": run_id}}
        try:
            state     = graph.get_state(config)
            state_uid = (state.values or {}).get("user_id") if hasattr(state, "values") else None
            if state_uid and state_uid != user_id:
                raise HTTPException(status_code=403, detail="Not your run")
        except HTTPException:
            raise
        except Exception:
            pass


def _snap_to_interrupt_payload(graph: Any, run_id: str) -> dict | None:
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
    pg_host:       str = ""
    pg_port:       int = 5432
    pg_dbname:     str = ""
    pg_user:       str = ""
    pg_password:   str = ""
    parent_run_id: str = ""       # set for follow-up queries; injects parent narrative as context

    @field_validator("analysis_mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("", "ab_test", "general"):
            raise ValueError("analysis_mode must be 'ab_test', 'general', or '' (auto)")
        return v

    @field_validator("pg_port")
    @classmethod
    def _check_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("pg_port out of range")
        return v


class ResumeRequest(BaseModel):
    gate:  str
    value: dict


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
):
    t0 = time.perf_counter()
    await check_rate_limit(current_user["user_id"])

    task = _sanitise_task(req.task)

    if req.pg_host:
        _validate_pg_host(req.pg_host)

    graph  = _get_graph(request)
    run_id = str(uuid.uuid4())

    resolved_duckdb_path = ""
    if req.duckdb_path:
        resolved_duckdb_path = resolve_upload_path(req.duckdb_path, current_user["user_id"])

    # Extract parent narrative for follow-up context injection
    context_narrative = ""
    if req.parent_run_id:
        try:
            parent_config = {"configurable": {"thread_id": req.parent_run_id}}
            parent_state  = graph.get_state(parent_config)
            parent_values = parent_state.values if hasattr(parent_state, "values") else {}
            raw_narrative = (
                parent_values.get("final_narrative")
                or parent_values.get("narrative_draft", "")
            )
            context_narrative = raw_narrative[:2000] if raw_narrative else ""
        except Exception:
            logger.warning("Could not read parent run state for %s", req.parent_run_id)

    await start_run(
        graph,
        run_id,
        {
            "task":               task,
            "analysis_mode":      req.analysis_mode,
            "db_backend":         req.db_backend,
            "duckdb_path":        resolved_duckdb_path,
            "pg_host":            req.pg_host,
            "pg_port":            req.pg_port,
            "pg_dbname":          req.pg_dbname,
            "pg_user":            req.pg_user,
            "pg_password":        req.pg_password,
            "user_id":            current_user["user_id"],
            "run_id":             run_id,
            "context_narrative":  context_narrative,
        },
        user_id=current_user["user_id"],
    )

    logger.info("run.start user=%s run=%s mode=%s backend=%s latency_ms=%.0f",
                current_user["user_id"], run_id, req.analysis_mode, req.db_backend,
                (time.perf_counter() - t0) * 1000)
    return {"run_id": run_id}


@router.get("/runs/{run_id}/stream")
async def stream_run(
    run_id: str,
    request: Request,
    token:   str = Query(...),
    last_id: str = Query(default="$"),  # pass Last-Event-ID on reconnect
):
    current_user = _user_from_token_param(token)
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
            interrupt_payload = _snap_to_interrupt_payload(graph, run_id)
            if interrupt_payload is not None:
                gate    = interrupt_payload.get("gate", "unknown")
                expires = int(time.time()) + _GATE_TIMEOUT_SECS
                logger.info("run.gate (replay) run=%s gate=%s", run_id, gate)
                from ..run_manager import get_redis_client as _get_redis
                _r = _get_redis()
                if _r:
                    await _r.set(f"run:gate_deadline:{run_id}", expires,
                                 ex=_GATE_TIMEOUT_SECS + 60)
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

            interrupt_payload = _snap_to_interrupt_payload(graph, run_id)

            if interrupt_payload is not None:
                gate    = interrupt_payload.get("gate", "unknown")
                expires = int(time.time()) + _GATE_TIMEOUT_SECS
                logger.info("run.gate run=%s gate=%s expires=%s", run_id, gate, expires)

                # Store timeout in Redis so the resume endpoint can reject stale gates
                from ..run_manager import get_redis_client as _get_redis
                _r = _get_redis()
                if _r:
                    await _r.set(f"run:gate_deadline:{run_id}", expires,
                                 ex=_GATE_TIMEOUT_SECS + 60)

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
                final_state  = graph.get_state(config)
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

    # Reject resume if the gate window has expired
    from ..run_manager import get_redis_client as _get_redis
    _r = _get_redis()
    if _r:
        deadline_raw = await _r.get(f"run:gate_deadline:{run_id}")
        if deadline_raw is not None:
            deadline = int(deadline_raw)
            if time.time() > deadline:
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail="Gate expired — please start a new analysis",
                )

    logger.info("run.resume run=%s gate=%s user=%s", run_id, req.gate, current_user["user_id"])
    await resume_run(graph, run_id, req.value)
    return {"status": "ok"}


@router.get("/runs")
def list_runs(
    request: Request,
    limit: int = Query(default=10, le=100),
    current_user: dict = Depends(get_current_user),
):
    store = _get_memory_store(request)
    try:
        return store.get_all_runs(user_id=current_user["user_id"], limit=limit)
    except Exception as exc:
        logger.warning("list_runs failed: %s", exc)
        return []


@router.get("/runs/{run_id}/detail")
async def get_run_detail(
    run_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    graph = _get_graph(request)
    await _check_ownership(graph, run_id, current_user["user_id"])
    config = {"configurable": {"thread_id": run_id}}
    try:
        state  = graph.get_state(config)
        values = state.values if hasattr(state, "values") else {}
    except Exception:
        raise HTTPException(status_code=404, detail="Run state not found")
    return {
        "run_id":         run_id,
        "task":           values.get("task", ""),
        "narrative":      values.get("final_narrative") or values.get("narrative_draft", ""),
        "recommendation": values.get("recommendation", ""),
    }


@router.get("/runs/{run_id}/pdf")
async def get_pdf(
    run_id: str,
    request: Request,
    token:          str = Query(...),
    narrative:      str = Query(default=""),
    task:           str = Query(default=""),
    recommendation: str = Query(default=""),
):
    current_user = _user_from_token_param(token)
    graph        = _get_graph(request)
    await _check_ownership(graph, run_id, current_user["user_id"])
    try:
        from ..pdf import build_pdf
        pdf_bytes = build_pdf(task=task, narrative=narrative, recommendation=recommendation)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="datapilot-{run_id[:8]}.pdf"'},
        )
    except Exception as exc:
        logger.exception("PDF generation failed for run %s", run_id)
        raise HTTPException(status_code=500, detail=str(exc))
