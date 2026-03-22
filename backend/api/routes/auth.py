"""
backend/api/routes/auth.py

POST /auth/register   {username, email, password}    → tokens + user
POST /auth/login      {login, password}              → tokens + user
POST /auth/refresh    {refresh_token}                → {access_token}
POST /auth/logout     {refresh_token}                → {status: "ok"}
GET  /auth/me                                        → user info
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

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
)
from ..deps import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    verify_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    login: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


def _token_response(user) -> dict:
    return {
        "access_token":  create_access_token(user.user_id, user.username),
        "refresh_token": create_refresh_token(user.user_id),
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


@router.post("/refresh")
def refresh(req: RefreshRequest):
    user_id, _jti = verify_refresh_token(req.refresh_token)
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    # Issue a fresh access token; keep the same refresh token (rotation on logout only)
    return {"access_token": create_access_token(user.user_id, user.username)}


@router.post("/logout")
def logout(req: RefreshRequest):
    """
    Revoke the refresh token so it cannot be used to obtain new access tokens.
    The client should also discard both tokens from storage.
    """
    try:
        _user_id, jti = verify_refresh_token(req.refresh_token)
        revoke_token(jti)
    except HTTPException:
        pass  # already invalid/revoked — idempotent logout
    return {"status": "ok"}


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:    str
    password: str


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
def forgot_password(req: ForgotPasswordRequest):
    """
    Generate a reset token and email it to the user.
    Always returns 202 — never reveals whether the email exists.
    """
    from ..email import send_password_reset
    token = create_reset_token(req.email)
    if token:
        try:
            send_password_reset(req.email, token)
        except RuntimeError:
            # Email delivery failed — surface as 500 so the user knows to retry
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not send reset email. Please try again.",
            )
    # Return 202 regardless of whether the email exists (prevents user enumeration)
    return {"detail": "If that email is registered, a reset link has been sent."}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest):
    """Consume a reset token and set the new password."""
    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters.",
        )
    user_id = consume_reset_token(req.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset link is invalid or has expired.",
        )
    update_password(user_id, req.password)
    return {"detail": "Password updated. You can now sign in."}


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    user = get_user_by_id(current_user["user_id"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user.to_dict()
