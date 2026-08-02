"""
backend/api/crypto_secrets.py — At-rest encryption for connection secrets.

Uses Fernet (AES-128-CBC + HMAC) with a key derived from SECRET_KEY via
HKDF-SHA256.  Industry-standard pattern for encrypting DB passwords before
persisting them; never log plaintext secrets.

Override with CONNECTIONS_ENCRYPTION_KEY (url-safe base64 Fernet key) for
key rotation without changing SECRET_KEY.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)

_FERNET: Fernet | None = None
_FERNET_KEY_FINGERPRINT: str | None = None


def reset_fernet_cache() -> None:
    """Clear cached Fernet instance (tests / key rotation)."""
    global _FERNET, _FERNET_KEY_FINGERPRINT
    _FERNET = None
    _FERNET_KEY_FINGERPRINT = None


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet key from an arbitrary secret string."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"datapilot-connections-v1",
        info=b"connection-password-encryption",
    )
    raw = hkdf.derive(secret.encode("utf-8"))
    return base64.urlsafe_b64encode(raw)


def _get_fernet() -> Fernet:
    global _FERNET, _FERNET_KEY_FINGERPRINT

    override = os.getenv("CONNECTIONS_ENCRYPTION_KEY", "").strip()
    if override:
        fp = f"override:{override[:16]}"
        if _FERNET is not None and _FERNET_KEY_FINGERPRINT == fp:
            return _FERNET
        try:
            _FERNET = Fernet(override.encode("utf-8") if isinstance(override, str) else override)
            _FERNET_KEY_FINGERPRINT = fp
            return _FERNET
        except Exception as exc:
            raise RuntimeError(
                "CONNECTIONS_ENCRYPTION_KEY is not a valid Fernet key"
            ) from exc

    from .deps import SECRET_KEY
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY required to encrypt connection secrets")
    fp = f"secret:{hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:16]}"
    if _FERNET is not None and _FERNET_KEY_FINGERPRINT == fp:
        return _FERNET
    _FERNET = Fernet(_derive_fernet_key(SECRET_KEY))
    _FERNET_KEY_FINGERPRINT = fp
    return _FERNET


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret string; returns url-safe base64 ciphertext."""
    if plaintext is None:
        plaintext = ""
    token = _get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a Fernet token; raises ValueError on tamper/wrong key."""
    if not ciphertext:
        return ""
    try:
        return _get_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Failed to decrypt connection secret — key may have rotated") from exc

