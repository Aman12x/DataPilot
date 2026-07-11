"""
backend/api/routes/auth.py

POST /auth/register   {username, email, password}    → user (+ verification email)
POST /auth/verify-email {token}                     → user (+ HttpOnly cookies)
POST /auth/resend-verification {email}              → 202
POST /auth/login      {login, password}              → user (+ HttpOnly cookies)
POST /auth/guest                                       → user (+ HttpOnly cookie)
POST /auth/refresh                                     → ok (+ rotated cookies)
POST /auth/logout                                      → {status: "ok"}
GET  /auth/me                                        → user info
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from auth.store import (
    consume_verification_token,
    create_user,
    create_verification_token,
    get_session_version,
    get_user_by_email,
    get_user_by_id,
    mark_email_verified,
    revoke_token,
    create_reset_token,
    consume_reset_token,
    update_password,
    verify_user,
)
from ..auth_rate import check_auth_rate
from ..cookies import clear_auth_cookies, read_refresh_token, set_auth_cookies
from ..deps import (
    create_access_token,
    create_guest_access_token,
    create_refresh_token,
    get_current_user,
    resolve_refresh_token,
    verify_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_ENV = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("ENV", "development")
_IS_PRODUCTION = _ENV.lower() in ("production", "prod")
_RETURN_TOKENS = os.getenv(
    "AUTH_RETURN_TOKENS",
    "false" if _IS_PRODUCTION else "true",
).lower() in ("1", "true", "yes")


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    login: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str = ""


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=1)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


def _should_auto_verify() -> bool:
    """Skip email verification when explicitly enabled or email is not configured."""
    auto = os.getenv("AUTH_AUTO_VERIFY_EMAIL", "").lower() in ("1", "true", "yes")
    return auto or not os.getenv("RESEND_API_KEY", "")


def _auth_response(
    user_dict: dict,
    access_token: str,
    refresh_token: str | None = None,
    *,
    status_code: int = 200,
) -> JSONResponse:
    """Set HttpOnly cookies; optionally include tokens in JSON for tests/API clients."""
    body: dict = {"user": user_dict}
    if _RETURN_TOKENS:
        body["access_token"] = access_token
        if refresh_token:
            body["refresh_token"] = refresh_token
    resp = JSONResponse(content=body, status_code=status_code)
    set_auth_cookies(resp, access_token, refresh_token)
    return resp


def _issue_session(user, *, status_code: int = 200) -> JSONResponse:
    sv = get_session_version(user.user_id)
    access = create_access_token(user.user_id, user.username)
    refresh = create_refresh_token(user.user_id, session_version=sv)
    return _auth_response(user.to_dict(), access, refresh, status_code=status_code)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, request: Request):
    await check_auth_rate(request, bucket="register")
    auto_verify = _should_auto_verify()
    result = create_user(
        req.username,
        req.email,
        req.password,
        email_verified=auto_verify,
    )
    if isinstance(result, str):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result)

    if auto_verify:
        return _issue_session(result, status_code=status.HTTP_201_CREATED)

    from ..email import send_verification_email

    token = create_verification_token(result.user_id)
    email_sent = True
    try:
        send_verification_email(result.email, token)
    except RuntimeError:
        email_sent = False

    detail = (
        "Check your email for a verification link before signing in."
        if email_sent
        else "Account created, but we could not send the verification email. "
             "Use Resend verification or resend below once email is configured."
    )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "user": result.to_dict(),
            "verify_pending": True,
            "email_sent": email_sent,
            "detail": detail,
        },
    )


@router.post("/verify-email")
async def verify_email(req: VerifyEmailRequest, request: Request):
    await check_auth_rate(request, bucket="verify")
    user_id = consume_verification_token(req.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification link is invalid or has expired.",
        )
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return _issue_session(user)


@router.post("/resend-verification", status_code=status.HTTP_202_ACCEPTED)
async def resend_verification(req: ResendVerificationRequest, request: Request):
    await check_auth_rate(request, bucket="verify")
    from ..email import send_verification_email

    user = get_user_by_email(req.email)
    if user and not user.email_verified:
        token = create_verification_token(user.user_id)
        try:
            send_verification_email(user.email, token)
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not send verification email. Please try again.",
            )
    return {
        "detail": "If that email is registered and unverified, a new link has been sent.",
    }


@router.post("/login")
async def login(req: LoginRequest, request: Request):
    await check_auth_rate(request, bucket="login")
    user = verify_user(req.login, req.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Check your inbox or request a new verification link.",
        )
    return _issue_session(user)


@router.post("/guest")
async def guest_session(request: Request):
    """Create an ephemeral anonymous session with a unique user_id."""
    await check_auth_rate(request, bucket="guest")
    access, guest_id = create_guest_access_token()
    user = {
        "user_id": guest_id,
        "username": "Guest",
        "email": "",
        "created_at": "",
        "email_verified": True,
    }
    return _auth_response(user, access)


@router.post("/refresh")
async def refresh(req: RefreshRequest, request: Request):
    await check_auth_rate(request, bucket="refresh")
    token = resolve_refresh_token(request, req.refresh_token or None)
    user_id, jti, sv = verify_refresh_token(token)
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    revoke_token(jti)
    access = create_access_token(user.user_id, user.username)
    refresh_tok = create_refresh_token(user.user_id, session_version=sv)
    return _auth_response(user.to_dict(), access, refresh_tok)


@router.post("/logout")
def logout(req: RefreshRequest, request: Request):
    """
    Revoke the refresh token and clear auth cookies.
    """
    token = read_refresh_token(request) or (req.refresh_token or None)
    if token:
        try:
            _user_id, jti, _sv = verify_refresh_token(token)
            revoke_token(jti)
        except HTTPException:
            pass
    resp = JSONResponse(content={"status": "ok"})
    clear_auth_cookies(resp)
    return resp


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:    str
    password: str = Field(min_length=8, max_length=256)


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(req: ForgotPasswordRequest, request: Request):
    """
    Generate a reset token and email it to the user.
    Always returns 202 — never reveals whether the email exists.
    """
    await check_auth_rate(request, bucket="forgot")
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
async def reset_password(req: ResetPasswordRequest, request: Request):
    """Consume a reset token and set the new password."""
    await check_auth_rate(request, bucket="reset")
    user_id = consume_reset_token(req.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset link is invalid or has expired.",
        )
    if not update_password(user_id, req.password):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters and include a letter and a number.",
        )
    mark_email_verified(user_id)
    return {"detail": "Password updated. You can now sign in."}


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    if current_user["user_id"].startswith("guest-"):
        return {
            "user_id": current_user["user_id"],
            "username": current_user.get("username", "Guest"),
            "email": "",
            "created_at": "",
            "email_verified": True,
        }
    user = get_user_by_id(current_user["user_id"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user.to_dict()
