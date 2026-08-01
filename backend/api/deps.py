"""
backend/api/deps.py — FastAPI dependencies (JWT auth).

JWT signed with SECRET_KEY env var (HS256).
  access token:  1 hour
  refresh token: 30 days  (includes jti for revocation + session_version)
  stream token:  15 minutes (scoped to a single run_id)
  pdf token:     5 minutes  (scoped to a single run_id)
"""
from __future__ import annotations

import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import JWTError, jwt

from .cookies import read_access_token, read_refresh_token

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 30
STREAM_TOKEN_EXPIRE_MINUTES = 15
PDF_TOKEN_EXPIRE_MINUTES = 5

logger = logging.getLogger(__name__)

_ENV = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("ENV", "development")
_IS_PRODUCTION = _ENV.lower() in ("production", "prod")

# The key check applies to any real deployment, not just one named "production".
# _IS_PRODUCTION only matches ("production", "prod"), so a Railway service whose
# environment is called "staging" would otherwise sign tokens with a key nothing
# validated. Being on Railway at all is enough to demand a real key.
_IS_DEPLOYED = _IS_PRODUCTION or bool(os.getenv("RAILWAY_ENVIRONMENT"))

# HS256 is HMAC-SHA256: RFC 7518 §3.2 requires a key of at least 256 bits.
_MIN_SECRET_KEY_LENGTH = 32
# Rejects keys that pass the length check but carry no entropy ("aaaa…",
# "abababab…"). 32 random hex chars yield ~16 distinct; 8 is a floor, not a target.
_MIN_SECRET_KEY_UNIQUE_CHARS = 8

_PLACEHOLDER_SECRETS = frozenset({
    "change-me-to-a-long-random-string",   # shipped in .env.example
    "change-me", "changeme", "change_me",
    "secret", "secretkey", "secret-key", "secret_key",
    "password", "passw0rd", "letmein",
    "test", "testing", "dev", "development", "local",
    "your-secret-key", "your_secret_key", "your_secret_key_here",
    "supersecret", "super-secret", "super-secret-key",
    "todo", "fixme", "xxx", "asdf", "qwerty",
})


def validate_secret_key(key: str) -> list[str]:
    """Return the reasons `key` is unsuitable for signing JWTs; empty if it's fine."""
    problems: list[str] = []
    if len(key) < _MIN_SECRET_KEY_LENGTH:
        problems.append(
            f"must be at least {_MIN_SECRET_KEY_LENGTH} characters (got {len(key)})"
        )
    if key.strip().lower() in _PLACEHOLDER_SECRETS:
        problems.append("is a well-known placeholder value")
    distinct = len(set(key))
    if distinct < _MIN_SECRET_KEY_UNIQUE_CHARS:
        problems.append(f"has only {distinct} distinct characters, so it is not random")
    return problems


_HOW_TO_GENERATE = (
    'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
)

SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    if _IS_DEPLOYED:
        raise RuntimeError(f"SECRET_KEY must be set in {_ENV}. {_HOW_TO_GENERATE}")
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY not set — using a random key (sessions won't survive restarts)"
    )
else:
    _problems = validate_secret_key(SECRET_KEY)
    if _problems:
        _summary = "SECRET_KEY " + "; it ".join(_problems)
        if _IS_DEPLOYED:
            raise RuntimeError(f"{_summary}. {_HOW_TO_GENERATE}")
        logger.warning("%s. Acceptable locally, but would fail to boot when deployed.", _summary)

bearer_scheme = HTTPBearer(auto_error=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _encode(payload: dict[str, Any]) -> str:
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str, username: str) -> str:
    expire = _utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
        "type": "access",
    }
    return _encode(payload)


def create_guest_access_token() -> tuple[str, str]:
    """Return (access_token, guest_user_id) for an ephemeral anonymous session."""
    guest_id = f"guest-{uuid.uuid4()}"
    expire = _utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": guest_id,
        "username": "Guest",
        "exp": expire,
        "type": "access",
        "guest": True,
    }
    return _encode(payload), guest_id


def create_refresh_token(user_id: str, session_version: int = 0) -> str:
    expire = _utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "sv": session_version,
    }
    return _encode(payload)


def create_stream_token(user_id: str, run_id: str) -> str:
    expire = _utcnow() + timedelta(minutes=STREAM_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "stream",
        "run_id": run_id,
    }
    return _encode(payload)


def create_pdf_token(user_id: str, run_id: str) -> str:
    expire = _utcnow() + timedelta(minutes=PDF_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "pdf",
        "run_id": run_id,
    }
    return _encode(payload)


def _decode_token(token: str, expected_type: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    if payload.get("type") != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type"
        )
    return payload


def _resolve_access_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    cookie_token = read_access_token(request)
    if cookie_token:
        return cookie_token
    if credentials is not None:
        return credentials.credentials
    return None


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, str]:
    token = _resolve_access_token(request, credentials)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    payload = _decode_token(token, "access")
    return {"user_id": payload["sub"], "username": payload.get("username", "")}


def bootstrap_user_workspace(user_id: str) -> dict[str, str] | None:
    """
    Ensure a real user has a personal workspace and migrated resources.
    Returns {workspace_id, role, name} or None for guests.
    """
    if not user_id or user_id.startswith("guest-"):
        return None
    from auth.org_store import (
        ensure_personal_workspace,
        migrate_user_resources_to_workspace,
    )

    ws = ensure_personal_workspace(user_id)
    migrate_user_resources_to_workspace(user_id, ws.workspace_id)
    return {
        "workspace_id": ws.workspace_id,
        "role": ws.role,
        "name": ws.name,
    }


def resolve_workspace_id(
    request: Request,
    current_user: dict[str, str] = Depends(get_current_user),
) -> str | None:
    """
    Resolve active workspace from X-Workspace-Id (or query workspace_id).

    Guests → None (legacy personal path).
    Real users → header/query if member, else personal workspace (auto-created).
    """
    user_id = current_user["user_id"]
    if user_id.startswith("guest-"):
        return None

    from auth.org_store import get_membership

    header_ws = (request.headers.get("X-Workspace-Id") or "").strip()
    query_ws = (request.query_params.get("workspace_id") or "").strip()
    requested = header_ws or query_ws

    personal = bootstrap_user_workspace(user_id)
    if not personal:
        return None

    if requested:
        role = get_membership(user_id, requested)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not a member of this workspace",
            )
        return requested
    return personal["workspace_id"]


def require_workspace_owner(
    request: Request,
    current_user: dict[str, str] = Depends(get_current_user),
) -> str:
    """Resolve workspace and require owner role. Returns workspace_id."""
    from auth.org_store import require_role

    ws_id = resolve_workspace_id(request, current_user)
    if not ws_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace required",
        )
    try:
        require_role(current_user["user_id"], ws_id, min_role="owner")
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    return ws_id


def resolve_refresh_token(
    request: Request,
    body_token: str | None = None,
) -> str:
    """Read refresh token from HttpOnly cookie or JSON body (API clients)."""
    token = read_refresh_token(request) or body_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    return token


def verify_refresh_token(refresh_token: str) -> tuple[str, str, int]:
    """
    Return (user_id, jti, session_version) from a valid, non-revoked refresh token.
    Raises 401 if the token is invalid, expired, or has been revoked.
    """
    payload = _decode_token(refresh_token, "refresh")
    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token must be re-issued"
        )
    from auth.store import get_session_version, is_token_revoked

    if is_token_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked"
        )
    user_id = payload["sub"]
    token_sv = int(payload.get("sv", 0))
    if token_sv != get_session_version(user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked"
        )
    return user_id, jti, token_sv


def verify_scoped_token(token: str, expected_type: str, run_id: str) -> dict[str, str]:
    """Validate a short-lived stream/pdf token scoped to run_id."""
    payload = _decode_token(token, expected_type)
    if payload.get("run_id") != run_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Token not valid for this run"
        )
    return {"user_id": payload["sub"], "username": ""}
