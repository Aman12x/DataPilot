"""
backend/api/cookies.py — HttpOnly auth cookie helpers.

Access and refresh tokens are stored in HttpOnly cookies so XSS cannot
exfiltrate session credentials. Authorization: Bearer remains supported
for tests and API clients.
"""
from __future__ import annotations

import os

from fastapi import Request, Response

ACCESS_COOKIE = "dp_access"
REFRESH_COOKIE = "dp_refresh"

_ENV = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("ENV", "development")
_IS_PRODUCTION = _ENV.lower() in ("production", "prod")

ACCESS_MAX_AGE = 60 * 60          # 1 hour
REFRESH_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _secure() -> bool:
    return _IS_PRODUCTION


def _samesite() -> str:
    # Cross-origin Railway deploys (frontend ↔ backend) need None + Secure.
    return "none" if _IS_PRODUCTION else "lax"


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str | None = None,
) -> None:
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        secure=_secure(),
        samesite=_samesite(),
        max_age=ACCESS_MAX_AGE,
        path="/",
    )
    if refresh_token:
        response.set_cookie(
            key=REFRESH_COOKIE,
            value=refresh_token,
            httponly=True,
            secure=_secure(),
            samesite=_samesite(),
            max_age=REFRESH_MAX_AGE,
            path="/",
        )


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/")


def read_access_token(request: Request) -> str | None:
    return request.cookies.get(ACCESS_COOKIE)


def read_refresh_token(request: Request) -> str | None:
    return request.cookies.get(REFRESH_COOKIE)
