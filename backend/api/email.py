"""
backend/api/email.py — Email delivery via Resend.

Requires env vars:
  RESEND_API_KEY   — from resend.com dashboard
  EMAIL_FROM       — verified sender address, e.g. "DataPilot <noreply@yourdomain.com>"
  APP_URL          — frontend public URL, e.g. https://datapilot.up.railway.app

When RESEND_API_KEY is not set, emails are logged to stderr instead of sent.
This keeps the dev workflow functional without an email account.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_API_KEY   = os.getenv("RESEND_API_KEY", "")
_FROM      = os.getenv("EMAIL_FROM", "DataPilot <noreply@datapilot.app>")
_APP_URL   = os.getenv("APP_URL", "http://localhost:5173")


def send_password_reset(to_email: str, token: str) -> None:
    """
    Send a password-reset email containing a one-time link.
    The link expires in 1 hour (enforced server-side).
    """
    reset_url = f"{_APP_URL.rstrip('/')}/reset-password?token={token}"
    subject   = "Reset your DataPilot password"
    html      = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:32px 24px;background:#1e1e2e;color:#cdd6f4;border-radius:12px;">
      <div style="text-align:center;margin-bottom:24px;">
        <span style="font-size:28px;color:#89b4fa;">✦</span>
        <span style="font-size:20px;font-weight:700;color:#cdd6f4;margin-left:8px;">DataPilot</span>
      </div>
      <h2 style="color:#cdd6f4;font-size:18px;margin-bottom:12px;">Reset your password</h2>
      <p style="color:#a6adc8;line-height:1.6;margin-bottom:24px;">
        Someone requested a password reset for your DataPilot account.
        Click the button below to set a new password. This link expires in <strong style="color:#cdd6f4;">1 hour</strong>.
      </p>
      <a href="{reset_url}"
         style="display:inline-block;padding:12px 28px;background:linear-gradient(135deg,#89b4fa,#74c7ec);
                color:#1e1e2e;font-weight:700;border-radius:8px;text-decoration:none;font-size:15px;">
        Reset password →
      </a>
      <p style="color:#585b70;font-size:12px;margin-top:24px;">
        If you didn't request this, you can safely ignore this email.
        Your password won't change until you click the link above.
      </p>
      <p style="color:#45475a;font-size:11px;margin-top:8px;word-break:break-all;">
        Or paste this URL: {reset_url}
      </p>
    </div>
    """

    if not _API_KEY:
        logger.warning(
            "RESEND_API_KEY not set — password reset email NOT sent to %s. "
            "Reset URL (dev only): %s",
            to_email, reset_url,
        )
        return

    try:
        import resend
        resend.api_key = _API_KEY
        resend.Emails.send({
            "from":    _FROM,
            "to":      [to_email],
            "subject": subject,
            "html":    html,
        })
        logger.info("Password reset email sent to %s", to_email)
    except Exception as exc:
        # Log but don't leak email address or token in the error response
        logger.exception("Failed to send password reset email: %s", exc)
        raise RuntimeError("Email delivery failed") from exc
