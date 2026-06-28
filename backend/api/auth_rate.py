"""
backend/api/auth_rate.py — IP-based rate limiting for auth endpoints.

Uses Redis ZSET when available (via run_manager client), otherwise in-memory deque.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque

from fastapi import HTTPException, Request, status

_local_rate: dict[str, Deque[float]] = {}


def _limits() -> tuple[int, int]:
    window = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))
    max_attempts = int(os.getenv("AUTH_RATE_MAX_ATTEMPTS", "10"))
    return window, max_attempts


def client_ip(request: Request) -> str:
    """Resolve client IP, honoring X-Forwarded-For from a trusted reverse proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def check_auth_rate(request: Request, *, bucket: str = "auth") -> None:
    """
    Enforce per-IP rate limit for auth endpoints.

    bucket: separate counters per endpoint group (auth, guest, verify).
    """
    window, max_attempts = _limits()
    ip = client_ip(request)
    key = f"{bucket}:{ip}"

    from .run_manager import get_redis_client

    redis = get_redis_client()
    if redis:
        now = time.time()
        rkey = f"auth_rate:{key}"
        win_start = now - window
        pipe = redis.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", win_start)
        pipe.zadd(rkey, {str(now): now})
        pipe.zcard(rkey)
        pipe.expire(rkey, window + 10)
        results = await pipe.execute()
        count = results[2]
        if count > max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many attempts. Please wait {window} seconds.",
            )
        return

    now = time.monotonic()
    dq = _local_rate.setdefault(key, deque())
    while dq and dq[0] < now - window:
        dq.popleft()
    if len(dq) >= max_attempts:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many attempts. Please wait {window} seconds.",
        )
    dq.append(now)


def reset_auth_rate_limits() -> None:
    """Clear in-memory auth rate-limit state (tests only)."""
    _local_rate.clear()
