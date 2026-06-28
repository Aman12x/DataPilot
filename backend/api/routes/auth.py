"""
backend/api/routes/auth.py

POST /auth/register   {username, email, password}    → tokens + user
POST /auth/login      {login, password}              → tokens + user
POST /auth/guest                                       → {access_token, user}
POST /auth/refresh    {refresh_token}                → {access_token, refresh_token}
POST /auth/logout     {refresh_token}                → {status: "ok"}
GET  /auth/me                                        → user info
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

# ── Auth-endpoint rate limiter ─────────────────────────────────────────────────
# 10 attempts per 60 s per IP — prevents brute-force on login/register.
_AUTH_WINDOW = 60          # seconds
_AUTH_MAX    = 10          # max attempts per window
_auth_rate: dict[str, Deque[float]] = {}


def _check_auth_rate(request: Request) -> None:
    ip  = request.client.host if request.client else "unknown"
    now = time.monotonic()
    dq  = _auth_rate.setdefault(ip, deque())
    while dq and dq[0] < now - _AUTH_WINDOW:
        dq.popleft()
    if len(dq) >= _AUTH_MAX:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many attempts. Please wait {_AUTH_WINDOW} seconds.",
        )
    dq.append(now)

from auth.store import (
    create_user, verify_user, get_user_by_id,
    revoke_token, create_reset_token, consume_reset_token, update_password,
    get_session_version,
)
from ..deps import (
    create_access_token,
    create_guest_access_token,
    create_refresh_token,
    get_current_user,
    verify_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    login: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str


def _token_response(user) -> dict:
    sv = get_session_version(user.user_id)
    return {
        "access_token":  create_access_token(user.user_id, user.username),
        "refresh_token": create_refresh_token(user.user_id, session_version=sv),
        "user":          user.to_dict(),
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, request: Request):
    _check_auth_rate(request)
    result = create_user(req.username, req.email, req.password)
    if isinstance(result, str):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result)
    return _token_response(result)


@router.post("/login")
def login(req: LoginRequest, request: Request):
    _check_auth_rate(request)
    user = verify_user(req.login, req.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return _token_response(user)


@router.post("/guest")
def guest_session():
    """Create an ephemeral anonymous session with a unique user_id."""
    access, guest_id = create_guest_access_token()
    return {
        "access_token": access,
        "user": {
            "user_id": guest_id,
            "username": "Guest",
            "email": "",
            "created_at": "",
        },
    }


@router.post("/refresh")
def refresh(req: RefreshRequest, request: Request):
    _check_auth_rate(request)
    user_id, jti, sv = verify_refresh_token(req.refresh_token)
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    revoke_token(jti)
    return {
        "access_token":  create_access_token(user.user_id, user.username),
        "refresh_token": create_refresh_token(user.user_id, session_version=sv),
    }


@router.post("/logout")
def logout(req: RefreshRequest):
    """
    Revoke the refresh token so it cannot be used to obtain new access tokens.
    The client should also discard both tokens from storage.
    """
    try:
        _user_id, jti, _sv = verify_refresh_token(req.refresh_token)
        revoke_token(jti)
    except HTTPException:
        pass  # already invalid/revoked — idempotent logout
    return {"status": "ok"}


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:    str
    password: str = Field(min_length=8, max_length=256)


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
def forgot_password(req: ForgotPasswordRequest, request: Request):
    """
    Generate a reset token and email it to the user.
    Always returns 202 — never reveals whether the email exists.
    """
    _check_auth_rate(request)
    from ..email import send_password_reset
    token = create_reset_token(req.email)
    if token:
        try:
            send_password_reset(req.email, token)
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not send reset email. Please try again.",
            )
    return {"detail": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, request: Request):
    """Consume a reset token and set the new password."""
    _check_auth_rate(request)
    user_id = consume_reset_token(req.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset link is invalid or has expired.",
        )
    if not update_password(user_id, req.password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters.",
        )
    return {"detail": "Password updated. You can now sign in."}


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    if current_user["user_id"].startswith("guest-"):
        return {
            "user_id": current_user["user_id"],
            "username": current_user.get("username", "Guest"),
            "email": "",
            "created_at": "",
        }
    user = get_user_by_id(current_user["user_id"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user.to_dict()
