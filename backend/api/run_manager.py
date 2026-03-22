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
import json
import logging
import os
import time
from collections import deque
from typing import Any

from langgraph.types import Command

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")

# ── In-memory fallback ────────────────────────────────────────────────────────
_queues:     dict[str, asyncio.Queue] = {}
_run_owners: dict[str, str]           = {}

# ── Redis client (injected from lifespan) ─────────────────────────────────────
_redis: Any = None

_STREAM_TTL  = 6 * 60 * 60   # 6 h
_OWNER_TTL   = 48 * 60 * 60  # 48 h

_MAX_RUNS    = int(os.getenv("MAX_RUNS_PER_WINDOW", "5"))
_WINDOW_SECS = int(os.getenv("RATE_WINDOW_SECONDS", "300"))

_local_rate: dict[str, deque] = {}


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


# Sync fallback used by ownership check in routes (before async path)
def get_run_owner(run_id: str) -> str | None:
    return _run_owners.get(run_id)


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


def cleanup_run(run_id: str) -> None:
    _queues.pop(run_id, None)
    _run_owners.pop(run_id, None)


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


# ── Graph invocation ──────────────────────────────────────────────────────────

async def start_run(graph: Any, run_id: str, initial_state: dict, user_id: str) -> None:
    await set_owner(run_id, user_id)
    if not _redis:
        _queues[run_id] = asyncio.Queue()
    asyncio.create_task(_invoke(graph, initial_state, run_id))


async def resume_run(graph: Any, run_id: str, resume_value: Any) -> None:
    if not _redis and run_id not in _queues:
        _queues[run_id] = asyncio.Queue()
    asyncio.create_task(_invoke(graph, Command(resume=resume_value), run_id))


_INVOKE_TIMEOUT = int(os.getenv("GRAPH_INVOKE_TIMEOUT", "600"))  # 10 min default


async def _invoke(graph: Any, arg: Any, run_id: str) -> None:
    config = {"configurable": {"thread_id": run_id}}
    try:
        snap = await asyncio.wait_for(
            asyncio.to_thread(graph.invoke, arg, config),
            timeout=_INVOKE_TIMEOUT,
        )
        await _publish_result(run_id, {"ok": True, "snap": snap})
    except asyncio.TimeoutError:
        logger.error("Graph invoke timed out after %ds for run %s", _INVOKE_TIMEOUT, run_id)
        await _publish_result(run_id, {"ok": False, "error": f"Analysis timed out after {_INVOKE_TIMEOUT}s. Please try a simpler query."})
    except Exception as exc:
        logger.exception("Graph invoke failed for run %s", run_id)
        await _publish_result(run_id, {"ok": False, "error": str(exc)})
