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


_IS_PRODUCTION = (
    os.getenv("RAILWAY_ENVIRONMENT", "") or os.getenv("ENV", "")
).lower() in ("production", "prod")


def _trusted_hops() -> int:
    """How many reverse proxies sit in front of this app.

    Railway terminates one. Default to 0 in dev, where the app is reached
    directly and X-Forwarded-For is pure client input.
    """
    return int(os.getenv("TRUSTED_PROXY_HOPS", "1" if _IS_PRODUCTION else "0"))


def client_ip(request: Request) -> str:
    """Resolve the client IP, counting back from the right of X-Forwarded-For.

    Taking the leftmost entry trusts the client: anyone can send
    `X-Forwarded-For: 1.2.3.4` and be treated as a new IP, which defeats every
    per-IP limit. Each proxy *appends* the address it actually saw, so the Nth
    entry from the right is the last hop the client could not forge.
    """
    hops = _trusted_hops()
    if hops > 0:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            if parts:
                return parts[-min(hops, len(parts))]
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
