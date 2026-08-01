"""
backend/api/budget.py — LLM spend caps.

The run rate limit alone does not bound spend: it is keyed on user_id, and
POST /auth/guest mints a fresh guest-{uuid4} on demand, so a caller could reset
the counter indefinitely. Budgets are therefore keyed on a *scope* that a guest
cannot re-roll — their IP — and enforced against a global daily ceiling as well
as a per-scope one.

Spend is tracked per UTC day. Redis when available (shared across pods, expires
on its own), in-memory otherwise (single pod, pruned as days roll over).
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "__global__"

# In-memory fallback: {day: {scope: usd}}
_local: dict[str, dict[str, float]] = {}
_local_lock = threading.Lock()

_DAY_TTL_SECONDS = 48 * 60 * 60


def _limits() -> tuple[float, float, float]:
    """(global daily, per-user daily, per-guest daily) in USD. 0 disables a cap."""
    return (
        float(os.getenv("LLM_DAILY_BUDGET_USD", "50")),
        float(os.getenv("LLM_USER_DAILY_BUDGET_USD", "5")),
        float(os.getenv("LLM_GUEST_DAILY_BUDGET_USD", "0.50")),
    )


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def is_guest(user_id: str) -> bool:
    return user_id.startswith("guest-")


def scope_for(user_id: str, ip: str | None) -> str:
    """Budget key for a caller.

    Guests are keyed on IP: their user_id is minted fresh per /auth/guest call,
    so keying on it would let anyone reset their own budget at will.
    """
    if is_guest(user_id):
        return f"ip:{ip or 'unknown'}"
    return f"user:{user_id}"


def limit_for(scope: str) -> float:
    _, user_limit, guest_limit = _limits()
    return guest_limit if scope.startswith("ip:") else user_limit


# ── Storage ───────────────────────────────────────────────────────────────────

async def _get(day: str, scope: str) -> float:
    from .run_manager import get_redis_client

    redis = get_redis_client()
    if redis:
        raw = await redis.get(f"spend:{day}:{scope}")
        return float(raw) if raw else 0.0
    with _local_lock:
        return _local.get(day, {}).get(scope, 0.0)


async def _incr(day: str, scope: str, usd: float) -> None:
    from .run_manager import get_redis_client

    redis = get_redis_client()
    if redis:
        key = f"spend:{day}:{scope}"
        await redis.incrbyfloat(key, usd)
        await redis.expire(key, _DAY_TTL_SECONDS)
        return
    with _local_lock:
        bucket = _local.setdefault(day, {})
        bucket[scope] = bucket.get(scope, 0.0) + usd
        # Days only ever move forward; drop anything that is not current.
        for stale in [d for d in _local if d != day]:
            del _local[stale]


# ── Enforcement ───────────────────────────────────────────────────────────────

async def check_budget(user_id: str, ip: str | None) -> None:
    """Reject the request if the global or per-scope daily budget is spent.

    Checked before a run starts. A run already in flight is allowed to finish —
    stopping mid-analysis would bill for the tokens and deliver nothing.
    """
    global_limit, _, _ = _limits()
    day = _today()

    if global_limit > 0:
        spent = await _get(day, GLOBAL_SCOPE)
        if spent >= global_limit:
            logger.error(
                "budget.global_exhausted spent=%.4f limit=%.2f day=%s", spent, global_limit, day
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Daily analysis capacity reached. Please try again tomorrow.",
            )

    scope = scope_for(user_id, ip)
    scope_limit = limit_for(scope)
    if scope_limit > 0:
        spent = await _get(day, scope)
        if spent >= scope_limit:
            logger.warning("budget.scope_exhausted scope=%s spent=%.4f limit=%.2f", scope, spent, scope_limit)
            detail = (
                "Guest usage limit reached. Sign up for a full account to continue."
                if scope.startswith("ip:")
                else "Daily usage limit reached. Please try again tomorrow."
            )
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)


async def record_spend(scope: str, usd: float) -> None:
    """Add a completed run's cost to the global and per-scope daily totals."""
    if usd <= 0:
        return
    day = _today()
    try:
        await _incr(day, GLOBAL_SCOPE, usd)
        await _incr(day, scope, usd)
    except Exception:
        # Losing a spend record is bad but not worth failing a finished run over.
        logger.warning("budget.record_failed scope=%s usd=%.6f", scope, usd, exc_info=True)


async def spend_today(scope: str) -> float:
    return await _get(_today(), scope)


def reset_local_budget() -> None:
    """Clear in-memory spend state (tests only)."""
    with _local_lock:
        _local.clear()
