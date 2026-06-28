"""
backend/api/deps.py — FastAPI dependencies (JWT auth).

JWT signed with SECRET_KEY env var (HS256).
  access token:  1 hour
  refresh token: 30 days  (includes jti for revocation + session_version)
  stream token:  15 minutes (scoped to a single run_id)
  pdf token:     5 minutes  (scoped to a single run_id)
"""
from __future__ import annotations

import os
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

_ENV = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("ENV", "development")
_IS_PRODUCTION = _ENV.lower() in ("production", "prod")

SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    if _IS_PRODUCTION:
        raise RuntimeError("SECRET_KEY must be set in production")
    import secrets as _secrets
    import logging as _logging

    SECRET_KEY = _secrets.token_hex(32)
    _logging.getLogger(__name__).warning(
        "SECRET_KEY not set — using a random key (sessions won't survive restarts)"
    )

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
