"""
backend/api/cookies.py — HttpOnly auth cookie helpers.

Access and refresh tokens are stored in HttpOnly cookies so XSS cannot
exfiltrate session credentials. Authorization: Bearer remains supported
for tests and API clients.
"""
from __future__ import annotations

from fastapi import Request, Response

from .environment import is_deployed

ACCESS_COOKIE = "dp_access"
REFRESH_COOKIE = "dp_refresh"

ACCESS_MAX_AGE = 60 * 60          # 1 hour
REFRESH_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _secure() -> bool:
    return is_deployed()


def _samesite() -> str:
    # Cross-origin Railway deploys (frontend ↔ backend) need None + Secure.
    # SameSite=None is only honoured alongside Secure, so these two move together.
    return "none" if is_deployed() else "lax"


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
