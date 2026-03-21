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
