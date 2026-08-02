"""
backend/api/auth_rate.py — IP-based rate limiting for auth endpoints.

Uses Redis ZSET when available (via run_manager client), otherwise in-memory deque.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import time
from collections import deque
from typing import Deque

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# One-shot diagnostic: logs how the bucket key was derived. Off by default.
_DEBUG_IP = os.getenv("DEBUG_CLIENT_IP", "").lower() in ("1", "true", "yes")

_local_rate: dict[str, Deque[float]] = {}


def _limits() -> tuple[int, int]:
    window = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))
    max_attempts = int(os.getenv("AUTH_RATE_MAX_ATTEMPTS", "10"))
    return window, max_attempts


# Addresses that can only be infrastructure, never a real client. 100.64.0.0/10
# is CGNAT, which is what Railway's internal proxy hops use.
_INFRA_NETS = [
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8",
        "169.254.0.0/16", "100.64.0.0/10",
        "::1/128", "fc00::/7", "fe80::/10",
    )
]


def _is_infra(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # not an address at all — never use it as a bucket key
    return any(ip in net for net in _INFRA_NETS)


def _trusted_hops() -> int:
    """Explicit hop count, when an operator knows their exact topology.

    0 (the default) selects the scan strategy in client_ip instead, which does
    not need the number of hops to be known or stable.
    """
    return int(os.getenv("TRUSTED_PROXY_HOPS", "0"))


def client_ip(request: Request) -> str:
    """Resolve the client IP used to key every per-IP limit.

    Measured on the deployed app rather than assumed. Railway sends:

        X-Forwarded-For: <real client>, <railway edge>
        peer:            100.64.0.x        (CGNAT, varies per request)

    and it *replaces* any inbound X-Forwarded-For -- a request sent with
    `X-Forwarded-For: 9.9.9.9` arrived without that value anywhere. So the
    leftmost entry is both the real client and unforgeable here.

    Two earlier attempts failed because they reasoned from the generic proxy
    model instead of this evidence. Counting one hop from the right picked the
    CGNAT peer; skipping infrastructure ranges picked the Railway edge, which
    is *public* (152.233.47.65, 152.233.47.67, ...) and rotates. Both produced
    a fresh bucket per request, so 20 concurrent bad logins all returned 401
    and rate limiting did nothing at all.

    Leftmost is only safe when the edge strips inbound X-Forwarded-For, which
    is true for Railway and most managed platforms but not for a bare reverse
    proxy. TRUSTED_PROXY_HOPS pins an exact position for those.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            hops = _trusted_hops()
            if hops > 0:
                return parts[-min(hops, len(parts))]
            return parts[0]
    return request.client.host if request.client else "unknown"


async def check_auth_rate(request: Request, *, bucket: str = "auth") -> None:
    """
    Enforce per-IP rate limit for auth endpoints.

    bucket: separate counters per endpoint group (auth, guest, verify).
    """
    window, max_attempts = _limits()
    ip = client_ip(request)
    key = f"{bucket}:{ip}"
    if _DEBUG_IP:
        logger.info(
            "auth_rate.key bucket=%s ip=%s xff=%r peer=%s",
            bucket, ip,
            request.headers.get("X-Forwarded-For"),
            request.client.host if request.client else None,
        )

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
