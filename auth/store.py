"""
auth/store.py — User account store for DataPilot.

Uses a separate SQLite DB from the memory store so auth concerns
never touch analysis data.

Password hashing: PBKDF2-HMAC-SHA256 with a 32-byte random salt per user,
260,000 iterations. No third-party crypto dependencies.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


def _auth_db_path() -> str:
    return os.getenv("AUTH_DB_PATH", "memory/auth.db")


def _connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_db(path: str | None = None) -> None:
    path = path or _auth_db_path()
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                username   TEXT UNIQUE NOT NULL,
                email      TEXT UNIQUE NOT NULL,
                pwd_hash   TEXT NOT NULL,
                salt       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # Revoked refresh tokens — checked on every /auth/refresh call.
        # jti (JWT ID) is stored; tokens without a jti are treated as revoked
        # to prevent use of old tokens issued before this table existed.
        con.execute("""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti        TEXT PRIMARY KEY,
                revoked_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0
            )
        """)


def revoke_token(jti: str, path: str | None = None) -> None:
    """Mark a refresh token JTI as revoked."""
    path = path or _auth_db_path()
    init_db(path)
    ts = datetime.now(timezone.utc).isoformat()
    with _connect(path) as con:
        con.execute(
            "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at) VALUES (?, ?)",
            (jti, ts),
        )


def create_reset_token(email: str, path: str | None = None) -> str | None:
    """
    Create a 1-hour password-reset token for the given email.
    Returns the token string, or None if no account with that email exists.
    """
    from datetime import timedelta
    path = path or _auth_db_path()
    init_db(path)
    with _connect(path) as con:
        row = con.execute("SELECT user_id FROM users WHERE email = ?",
                          (email.strip().lower(),)).fetchone()
    if row is None:
        return None
    token      = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    with _connect(path) as con:
        con.execute(
            "INSERT INTO password_reset_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, row["user_id"], expires_at),
        )
    return token


def consume_reset_token(token: str, path: str | None = None) -> str | None:
    """
    Validate and consume a reset token.
    Returns user_id on success, None if invalid/expired/already used.
    Marks the token as used so it cannot be replayed.
    """
    path = path or _auth_db_path()
    init_db(path)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as con:
        row = con.execute(
            """SELECT user_id, expires_at, used FROM password_reset_tokens
               WHERE token = ?""",
            (token,),
        ).fetchone()
        if row is None or row["used"] or row["expires_at"] < now:
            return None
        con.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,)
        )
    return row["user_id"]


def update_password(user_id: str, new_password: str, path: str | None = None) -> bool:
    """Update the password for a user. Returns True on success."""
    if len(new_password) < 8:
        return False
    path = path or _auth_db_path()
    init_db(path)
    salt     = secrets.token_hex(32)
    pwd_hash = _hash_password(new_password, salt)
    with _connect(path) as con:
        con.execute(
            "UPDATE users SET pwd_hash = ?, salt = ? WHERE user_id = ?",
            (pwd_hash, salt, user_id),
        )
    return True


def is_token_revoked(jti: str, path: str | None = None) -> bool:
    """Return True if this JTI has been revoked."""
    path = path or _auth_db_path()
    init_db(path)
    with _connect(path) as con:
        row = con.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)
        ).fetchone()
    return row is not None


@dataclass
class User:
    user_id:    str
    username:   str
    email:      str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "user_id":    self.user_id,
            "username":   self.username,
            "email":      self.email,
            "created_at": self.created_at,
        }


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    )
    return dk.hex()


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_user(
    username: str,
    email: str,
    password: str,
    path: str | None = None,
) -> User | str:
    """
    Create a new user account.

    Returns:
        User on success.
        str (error message) on failure — duplicate username/email, weak password.
    """
    if len(password) < 8:
        return "Password must be at least 8 characters."

    path = path or _auth_db_path()
    init_db(path)

    user_id = str(uuid.uuid4())
    salt    = secrets.token_hex(32)
    pwd_hash = _hash_password(password, salt)
    ts      = datetime.now(timezone.utc).isoformat()

    try:
        with _connect(path) as con:
            con.execute(
                """INSERT INTO users (user_id, username, email, pwd_hash, salt, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, username.strip(), email.strip().lower(), pwd_hash, salt, ts),
            )
    except sqlite3.IntegrityError as exc:
        msg = str(exc)
        if "username" in msg:
            return f"Username '{username}' is already taken."
        if "email" in msg:
            return f"An account with email '{email}' already exists."
        return "Account creation failed. Please try again."

    return User(user_id=user_id, username=username.strip(), email=email.strip().lower(), created_at=ts)


def verify_user(
    login: str,
    password: str,
    path: str | None = None,
) -> User | None:
    """
    Verify credentials. `login` can be username or email.

    Returns User on success, None on failure.
    """
    path = path or _auth_db_path()
    init_db(path)

    with _connect(path) as con:
        row = con.execute(
            """SELECT * FROM users
               WHERE username = ? OR email = ?""",
            (login.strip(), login.strip().lower()),
        ).fetchone()

    if row is None:
        return None

    expected = _hash_password(password, row["salt"])
    if not secrets.compare_digest(expected, row["pwd_hash"]):
        return None

    return User(
        user_id=row["user_id"],
        username=row["username"],
        email=row["email"],
        created_at=row["created_at"],
    )


def get_user_by_id(user_id: str, path: str | None = None) -> User | None:
    path = path or _auth_db_path()
    init_db(path)
    with _connect(path) as con:
        row = con.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return User(
        user_id=row["user_id"],
        username=row["username"],
        email=row["email"],
        created_at=row["created_at"],
    )
