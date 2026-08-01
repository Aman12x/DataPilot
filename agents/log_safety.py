"""
agents/log_safety.py — keep user-supplied content out of log records.

Analyst tasks are business questions ("why did enterprise churn spike in EMEA")
and belong to the customer, not the operator. They were being logged verbatim at
INFO, and sentry-sdk turns INFO records into breadcrumbs by default, so anything
logged here is shipped to a third party the moment SENTRY_DSN is set.

Set LOG_USER_CONTENT=true to see the real text while debugging locally.
"""
from __future__ import annotations

import hashlib
import os
import secrets

# Per-process salt: digests correlate log lines within one process without
# being guessable. A short task would be trivially recoverable from an unsalted
# digest, so this is not a stable identifier across restarts by design.
_SALT = secrets.token_bytes(16)


def log_user_content_enabled() -> bool:
    """Read at call time so tests and running processes can toggle it."""
    return os.getenv("LOG_USER_CONTENT", "").strip().lower() in ("1", "true", "yes")


def redact(text: object, *, limit: int = 80) -> str:
    """Return a log-safe stand-in for user-supplied text.

    Yields `<redacted len=42 ref=1a2b3c4d>`: enough to tell entries apart and
    confirm something non-empty arrived, with no content. `ref` is salted per
    process — it correlates lines within one run, not across restarts.
    """
    if text is None:
        return "<none>"
    value = str(text)
    if not value:
        return "<empty>"
    if log_user_content_enabled():
        return value[:limit]
    digest = hashlib.sha256(_SALT + value.encode("utf-8", "replace")).hexdigest()[:8]
    return f"<redacted len={len(value)} ref={digest}>"
