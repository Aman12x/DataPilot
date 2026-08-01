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


_IS_PRODUCTION = (
    os.getenv("RAILWAY_ENVIRONMENT", "") or os.getenv("ENV", "")
).lower() in ("production", "prod")


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
    """Resolve the client IP from X-Forwarded-For, right to left.

    Taking the leftmost entry trusts the client: anyone can send
    `X-Forwarded-For: 1.2.3.4` and land in a fresh bucket, defeating every
    per-IP limit. Each proxy *appends* the address it saw, so the truth is on
    the right.

    A fixed hop count is not enough either. Railway's internal hop addresses
    come from 100.64.0.0/10 and differ per request, so pinning "one hop from
    the right" produced a new bucket every time and silently disabled rate
    limiting entirely -- 20 concurrent bad logins all returned 401, none 429.

    So: scan right to left and take the first address that could plausibly be a
    client. Infrastructure ranges are skipped, and anything a client forged
    sits to the left of the addresses the proxies appended, so it is never
    reached.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        hops = _trusted_hops()
        if hops > 0 and parts:
            return parts[-min(hops, len(parts))]
        for candidate in reversed(parts):
            if not _is_infra(candidate):
                return candidate
    if request.client and not _is_infra(request.client.host):
        return request.client.host
    # Everything looked internal: fall back to the peer so callers still share
    # a bucket rather than each getting an unlimited one.
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
