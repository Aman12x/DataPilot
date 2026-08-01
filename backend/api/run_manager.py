"""
backend/api/run_manager.py — per-run orchestration with optional Redis Streams.

When REDIS_URL is set:
  - Ownership stored in Redis (survives restarts, shared across pods).
  - Each invoke result is appended to a Redis Stream `run:{run_id}`.
  - SSE endpoint reads from the stream; entry IDs enable safe reconnects.
  - Rate-limit counters stored in Redis ZSET (shared across pods).

When REDIS_URL is not set (local dev / single-pod):
  - Falls back to in-memory asyncio.Queue + dict.
  - Behaviour identical to the original implementation.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.types import Command

from agents import spend

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")

# ── In-memory fallback ────────────────────────────────────────────────────────
_queues:     dict[str, asyncio.Queue] = {}
_run_owners: dict[str, str]           = {}
_run_scopes: dict[str, str]           = {}
_run_errors: OrderedDict[str, str]    = OrderedDict()  # run_id → error (capped, survives cleanup)
_RUN_ERRORS_MAX = 1000

# ── Redis client (injected from lifespan) ─────────────────────────────────────
_redis: Any = None

_STREAM_TTL  = 6 * 60 * 60   # 6 h
_OWNER_TTL   = 48 * 60 * 60  # 48 h

_MAX_RUNS    = int(os.getenv("MAX_RUNS_PER_WINDOW", "5"))
_WINDOW_SECS = int(os.getenv("RATE_WINDOW_SECONDS", "300"))

_local_rate: dict[str, deque] = {}
_gate_deadlines: dict[str, int] = {}
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_GRAPH_INVOKES", "32"))
_GATE_TIMEOUT_SECS = int(os.getenv("GATE_TIMEOUT_SECONDS", str(4 * 60 * 60)))
_active_invokes = 0

# Strong references to in-flight invoke tasks. asyncio only holds weak
# references, so without this a running task can be garbage-collected mid-run.
_active_tasks: set[asyncio.Task] = set()
_SHUTDOWN_TIMEOUT = float(os.getenv("SHUTDOWN_CANCEL_TIMEOUT", "10"))

# Graph execution gets its own thread pool rather than asyncio's default
# executor. The default is shared process-wide and sized min(32, cpu+4) — on a
# 2-vCPU box that is 6 threads, so unrelated to_thread() callers could occupy
# every slot and block all analyses. Sizing this to _MAX_CONCURRENT also makes
# that admission cap honest: previously it allowed 32 concurrent invokes on a
# pool that could only ever run 6.
_graph_executor: ThreadPoolExecutor | None = None


def _get_graph_executor() -> ThreadPoolExecutor:
    global _graph_executor
    if _graph_executor is None:
        _graph_executor = ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT,
            thread_name_prefix="graph-invoke",
        )
    return _graph_executor


def set_redis_client(client: Any) -> None:
    global _redis
    _redis = client


def get_redis_client() -> Any:
    return _redis


# ── Ownership ─────────────────────────────────────────────────────────────────

async def set_owner(run_id: str, user_id: str) -> None:
    if _redis:
        await _redis.set(f"run:owner:{run_id}", user_id, ex=_OWNER_TTL)
    else:
        _run_owners[run_id] = user_id


async def get_owner(run_id: str) -> str | None:
    if _redis:
        val = await _redis.get(f"run:owner:{run_id}")
        return val.decode() if val else None
    return _run_owners.get(run_id)


async def set_budget_scope(run_id: str, scope: str) -> None:
    """Remember which budget a run's spend bills to, so gate resumes bill it too."""
    if _redis:
        await _redis.set(f"run:budget:{run_id}", scope, ex=_OWNER_TTL)
    else:
        _run_scopes[run_id] = scope


async def get_budget_scope(run_id: str) -> str | None:
    if _redis:
        val = await _redis.get(f"run:budget:{run_id}")
        return val.decode() if val else None
    return _run_scopes.get(run_id)



# ── Result stream ─────────────────────────────────────────────────────────────

async def _publish_result(run_id: str, payload: dict) -> None:
    if _redis:
        key = f"run:{run_id}"
        await _redis.xadd(key, {"data": json.dumps(payload)})
        await _redis.expire(key, _STREAM_TTL)
    else:
        q = _queues.get(run_id)
        if q is not None:
            await q.put(payload)


async def read_result(run_id: str, last_id: str = "$") -> dict | None:
    """
    Block-read one result.

    last_id="$"  → wait for the next new entry (initial SSE connect).
    last_id="0"  → read from stream beginning (reconnect after server restart).
    last_id="<id>" → read entries after this ID (mid-stream reconnect).

    Returns None on timeout; caller sends a keepalive and retries.

    The in-memory path awaits an asyncio.Queue directly. It must never block a
    worker thread: one SSE viewer per connected client holding a thread for the
    full 30s poll would starve the pool that graph execution runs on.
    """
    if _redis:
        reply = await _redis.xread(
            {f"run:{run_id}": last_id},
            block=30_000,
            count=1,
        )
        if not reply:
            return None
        _key, entries = reply[0]
        entry_id, fields = entries[0]
        data = json.loads(fields[b"data"])
        data["_stream_id"] = entry_id.decode()
        return data
    else:
        q = _queues.get(run_id)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=30.0)
        except asyncio.TimeoutError:
            return None


def cleanup_run(run_id: str, *, drop_owner: bool = True) -> None:
    _queues.pop(run_id, None)
    if drop_owner:
        _run_owners.pop(run_id, None)


def _cache_error(run_id: str, msg: str) -> None:
    """Write to _run_errors, evicting oldest entry when cap is reached."""
    _run_errors[run_id] = msg
    while len(_run_errors) > _RUN_ERRORS_MAX:
        _run_errors.popitem(last=False)  # evict oldest


async def get_cached_error(run_id: str) -> str | None:
    """Return a cached error for run_id from Redis (if available) or memory."""
    if _redis:
        val = await _redis.get(f"run:error:{run_id}")
        return val.decode() if val else None
    return _run_errors.get(run_id)


# ── Rate limiting ─────────────────────────────────────────────────────────────

async def check_rate_limit(user_id: str) -> None:
    from fastapi import HTTPException, status as st

    if _redis:
        now    = time.time()
        key    = f"rate:{user_id}"
        window = now - _WINDOW_SECS
        pipe   = _redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, _WINDOW_SECS + 10)
        results = await pipe.execute()
        count   = results[2]
        if count > _MAX_RUNS:
            raise HTTPException(
                status_code=st.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit: max {_MAX_RUNS} runs per {_WINDOW_SECS}s",
            )
    else:
        now = time.monotonic()
        dq  = _local_rate.setdefault(user_id, deque())
        while dq and dq[0] < now - _WINDOW_SECS:
            dq.popleft()
        if len(dq) >= _MAX_RUNS:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit: max {_MAX_RUNS} runs per {_WINDOW_SECS}s",
            )
        dq.append(now)


# ── Node labels for Chain-of-Thought streaming ────────────────────────────────

_NODE_LABELS: dict[str, str | None] = {
    "check_semantic_cache": "Checking semantic cache",
    "inject_history":       "Loading analyst history",
    "load_schema":          "Reading database schema",
    "resolve_task_intent":  "Interpreting task",
    "infer_metric_config":  "Detecting metrics",
    "generate_sql":         "Writing SQL query",
    "execute_query":        "Executing query",
    "load_auxiliary_data":  "Loading auxiliary data",
    "decompose_metric":     "Decomposing metric",
    "detect_anomaly":       "Checking for anomalies",
    "forecast_baseline":    "Forecasting baseline",
    "run_cuped":            "CUPED variance reduction",
    "run_ttest":            "Running t-test",
    "check_srm":            "Checking sample ratio",
    "run_hte":              "HTE subgroup analysis",
    "detect_novelty":       "Novelty effect check",
    "compute_mde":          "Computing MDE",
    "check_guardrails":     "Checking guardrails",
    "compute_funnel":       "Funnel analysis",
    "describe_data":        "Describing data",
    "find_correlations":    "Computing correlations",
    "run_regression":       "Running regression",
    "detect_timeseries":    "Detecting time series",
    "generate_charts":      "Generating charts",
    "generate_narrative":   "Writing narrative",
    "log_run":              "Logging run",
    "run_power_analysis":   "Power analysis",
    # Gate nodes → None (skip)
    "query_gate": None, "analysis_gate": None, "narrative_gate": None,
    "semantic_cache_gate": None, "intent_gate": None,
}


def _step_detail(node: str, delta: dict) -> str | None:
    """Extract a one-liner detail from a node's state delta."""
    try:
        if node == "execute_query":
            qr = delta.get("query_result")
            if qr is not None and hasattr(qr, "__len__"):
                return f"{len(qr):,} rows returned"
        elif node == "run_cuped":
            r = delta.get("cuped_result")
            if r is not None:
                pct = getattr(r, "variance_reduction_pct", None)
                if pct is not None:
                    return f"Variance reduced {pct:.0f}%"
        elif node == "run_ttest":
            r = delta.get("ttest_result")
            if r is not None:
                p = getattr(r, "p_value", None)
                sig = getattr(r, "significant", None)
                if p is not None:
                    label = "significant" if sig else "not significant"
                    return f"p={p:.4f} ({label})"
        elif node == "check_srm":
            r = delta.get("srm_result")
            if r is not None:
                return "SRM detected" if getattr(r, "srm_detected", False) else "No SRM"
        elif node == "run_hte":
            r = delta.get("hte_result")
            if r is not None:
                seg = getattr(r, "top_segment", None)
                if seg:
                    return f"Top segment: {seg}"
        elif node == "run_regression":
            r = delta.get("regression_result")
            if r is not None:
                r2 = getattr(r, "r_squared", None)
                nf = getattr(r, "n_features", None)
                if r2 is not None:
                    return f"R²={r2:.3f}, {nf} predictors"
        elif node == "detect_anomaly":
            r = delta.get("anomaly_result")
            if r is not None:
                dates = getattr(r, "anomaly_dates", None)
                if dates is not None:
                    return f"{len(dates)} anomaly dates" if dates else "No anomalies"
        elif node == "check_guardrails":
            r = delta.get("guardrail_result")
            if r is not None:
                bc = getattr(r, "breached_count", None)
                if bc is not None:
                    return f"{bc} guardrail(s) breached" if bc else "All clear"
        elif node == "generate_charts":
            charts = delta.get("charts")
            if charts is not None:
                return f"{len(charts)} chart(s) generated"
        elif node == "describe_data":
            r = delta.get("describe_result")
            if r is not None:
                rc = getattr(r, "row_count", None)
                cc = getattr(r, "col_count", None)
                if rc is not None:
                    return f"{rc:,} rows, {cc} columns"
        elif node == "find_correlations":
            r = delta.get("correlation_result")
            if r is not None:
                pairs = getattr(r, "pairs", None)
                if pairs is not None:
                    return f"{len(pairs)} correlation pairs"
    except Exception:
        pass
    return None


async def set_gate_deadline(run_id: str, expires: int) -> None:
    if _redis:
        await _redis.set(f"run:gate_deadline:{run_id}", expires, ex=_GATE_TIMEOUT_SECS + 60)
    else:
        _gate_deadlines[run_id] = expires


async def get_gate_deadline(run_id: str) -> int | None:
    if _redis:
        deadline_raw = await _redis.get(f"run:gate_deadline:{run_id}")
        return int(deadline_raw) if deadline_raw is not None else None
    return _gate_deadlines.get(run_id)


def _spawn_invoke(graph: Any, arg: Any, run_id: str) -> asyncio.Task:
    task = asyncio.create_task(_invoke(graph, arg, run_id))
    _active_tasks.add(task)
    task.add_done_callback(_active_tasks.discard)
    return task


async def cancel_active_runs() -> None:
    """Cancel in-flight invoke tasks so the process can shut down cleanly.

    Cancelling the coroutine does not stop the worker thread already inside
    graph.stream() — it only releases the event loop so shutdown can proceed.
    """
    global _graph_executor

    tasks = [t for t in _active_tasks if not t.done()]
    if tasks:
        logger.info("Cancelling %d in-flight run task(s)", len(tasks))
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_SHUTDOWN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out after %.0fs waiting for run tasks to cancel",
                _SHUTDOWN_TIMEOUT,
            )

    if _graph_executor is not None:
        # wait=False: a thread already inside graph.stream() cannot be
        # interrupted, so blocking here would stall shutdown for up to
        # GRAPH_INVOKE_TIMEOUT. Queued-but-unstarted work is dropped.
        _graph_executor.shutdown(wait=False, cancel_futures=True)
        _graph_executor = None


async def start_run(
    graph: Any,
    run_id: str,
    initial_state: dict,
    user_id: str,
    budget_scope: str | None = None,
) -> None:
    await set_owner(run_id, user_id)
    await set_budget_scope(run_id, budget_scope or f"user:{user_id}")
    if not _redis:
        _queues[run_id] = asyncio.Queue()
    _spawn_invoke(graph, initial_state, run_id)


async def resume_run(graph: Any, run_id: str, resume_value: Any) -> None:
    if not _redis and run_id not in _queues:
        _queues[run_id] = asyncio.Queue()
    _spawn_invoke(graph, Command(resume=resume_value), run_id)


_INVOKE_TIMEOUT = int(os.getenv("GRAPH_INVOKE_TIMEOUT", "600"))  # 10 min default
_ERROR_TTL      = 6 * 60 * 60  # 6 h — same as stream TTL


async def _store_error(run_id: str, msg: str) -> None:
    """Persist error so reconnecting SSE clients get it immediately.
    Falls back to in-memory cache if Redis write fails."""
    if _redis:
        try:
            await _redis.set(f"run:error:{run_id}", msg, ex=_ERROR_TTL)
            return
        except Exception:
            logger.warning("Redis error cache write failed for run %s — falling back to memory", run_id)
    _cache_error(run_id, msg)


def _stream_graph(
    graph: Any,
    arg: Any,
    config: dict,
    run_id: str,
    loop: asyncio.AbstractEventLoop,
    run_meter: spend.Meter,
) -> dict:
    """Stream graph execution in a worker thread, publishing step events.

    The meter is activated here, inside the worker thread, so concurrent runs
    accumulate into their own contexts. The caller owns the Meter object and can
    read the total even if this call times out.
    """
    with spend.meter(run_meter):
        return _run_graph_stream(graph, arg, config, run_id, loop)


def _run_graph_stream(
    graph: Any, arg: Any, config: dict, run_id: str, loop: asyncio.AbstractEventLoop
) -> dict:
    for chunk in graph.stream(arg, config, stream_mode="updates"):
        if not chunk:
            continue
        node = list(chunk.keys())[0]
        label = _NODE_LABELS.get(node)
        if label is None:
            continue  # skip gate nodes and unknown nodes
        detail = _step_detail(node, chunk[node])
        event: dict = {"type": "step", "node": node, "label": label, "status": "completed"}
        if detail is not None:
            event["detail"] = detail
        try:
            _publish_sync(run_id, event, loop)
        except Exception:
            logger.debug("step publish failed for run %s", run_id, exc_info=True)
    return graph.get_state(config).values


def _publish_sync(run_id: str, payload: dict, loop: asyncio.AbstractEventLoop) -> None:
    """Publish a step event from a worker thread onto the event loop.

    `loop` must be passed in: asyncio.get_event_loop() raises in a non-main
    thread that has no loop of its own, which previously made every Redis-mode
    step event fail silently and stripped the whole chain-of-thought stream.
    """
    if _redis:
        asyncio.run_coroutine_threadsafe(
            _publish_result(run_id, payload), loop
        ).result(timeout=2)
        return
    q = _queues.get(run_id)
    if q is not None:
        loop.call_soon_threadsafe(q.put_nowait, payload)


async def _bill_run(run_id: str, run_meter: spend.Meter) -> None:
    """Charge a finished run's LLM spend to the budget scope it started under."""
    if run_meter.total_usd <= 0:
        return
    try:
        from .budget import record_spend

        scope = await get_budget_scope(run_id)
        if scope is None:
            owner = await get_owner(run_id)
            scope = f"user:{owner}" if owner else "user:unknown"
        await record_spend(scope, run_meter.total_usd)
        logger.info(
            "run.spend run=%s scope=%s usd=%.6f calls=%d",
            run_id, scope, run_meter.total_usd, run_meter.calls,
        )
    except Exception:
        logger.warning("Could not record spend for run %s", run_id, exc_info=True)


async def _invoke(graph: Any, arg: Any, run_id: str) -> None:
    global _active_invokes
    if _active_invokes >= _MAX_CONCURRENT:
        msg = "Server is busy. Please try again in a moment."
        await _publish_result(run_id, {"ok": False, "error": msg})
        await _store_error(run_id, msg)
        cleanup_run(run_id)
        return

    _active_invokes += 1
    config = {"configurable": {"thread_id": run_id}}
    loop = asyncio.get_running_loop()
    run_meter = spend.Meter()
    try:
        snap = await asyncio.wait_for(
            loop.run_in_executor(
                _get_graph_executor(),
                functools.partial(_stream_graph, graph, arg, config, run_id, loop, run_meter),
            ),
            timeout=_INVOKE_TIMEOUT,
        )
        await _publish_result(run_id, {"ok": True, "snap": snap})
    except asyncio.TimeoutError:
        msg = f"Analysis timed out after {_INVOKE_TIMEOUT}s. Please try a simpler query."
        logger.error("Graph invoke timed out after %ds for run %s", _INVOKE_TIMEOUT, run_id)
        await _publish_result(run_id, {"ok": False, "error": msg})
        await _store_error(run_id, msg)
        cleanup_run(run_id, drop_owner=False)
    except Exception as exc:
        logger.exception("Graph invoke failed for run %s", run_id)
        msg = "Analysis failed. Please try again."
        await _publish_result(run_id, {"ok": False, "error": msg})
        await _store_error(run_id, msg)
        cleanup_run(run_id, drop_owner=False)
    finally:
        _active_invokes -= 1
        # Bill on every exit path. A run that failed or timed out still spent
        # whatever tokens it consumed before it stopped.
        await _bill_run(run_id, run_meter)
