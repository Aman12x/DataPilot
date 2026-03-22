"""
backend/api/deps.py — FastAPI dependencies (JWT auth).

JWT signed with SECRET_KEY env var (HS256).
  access token:  1 hour
  refresh token: 30 days  (includes jti for revocation)
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import JWTError, jwt

SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    import sys
    print("FATAL: SECRET_KEY env var is not set. Set it to a long random string.", file=sys.stderr)
    sys.exit(1)
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS   = 30

bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(user_id: str, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "username": username, "exp": expire, "type": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    # jti (JWT ID) enables per-token revocation on logout
    payload = {"sub": user_id, "exp": expire, "type": "refresh", "jti": str(uuid.uuid4())}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str, expected_type: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")
    return payload


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, str]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = _decode_token(credentials.credentials, "access")
    return {"user_id": payload["sub"], "username": payload.get("username", "")}


def verify_refresh_token(refresh_token: str) -> tuple[str, str]:
    """
    Return (user_id, jti) from a valid, non-revoked refresh token.
    Raises 401 if the token is invalid, expired, or has been revoked.
    """
    payload = _decode_token(refresh_token, "refresh")
    jti = payload.get("jti")
    if not jti:
        # Tokens issued before jti was added — treat as revoked
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token must be re-issued")
    from auth.store import is_token_revoked
    if is_token_revoked(jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")
    return payload["sub"], jti
