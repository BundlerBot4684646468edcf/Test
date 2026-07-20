"""Cryptographic helpers — no third-party dependencies.

The pass token that a customer carries (in a QR code / wallet) contains only an
identifier plus a signature. It never carries the authoritative balance, so a
tampered token simply fails signature verification and the real balance is read
from the server.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


def new_id(prefix: str = "") -> str:
    """Return a high-entropy, non-sequential identifier (128 bits).

    Non-enumerable ids defend against guessing/hijacking valid passes (T7).
    """
    token = secrets.token_urlsafe(16)
    return f"{prefix}{token}" if prefix else token


def new_redemption_code() -> str:
    """A single-use, high-entropy redemption nonce (T3)."""
    return secrets.token_urlsafe(12)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_pass(pass_id: str, secret: str) -> str:
    """Produce a tamper-proof token ``<pass_id>.<sig>`` (T1, T2).

    The signature covers the pass id with HMAC-SHA256. Any edit to the id
    invalidates the signature.
    """
    sig = hmac.new(secret.encode("utf-8"), pass_id.encode("utf-8"), hashlib.sha256).digest()
    return f"{pass_id}.{_b64(sig)}"


def verify_pass(token: str, secret: str) -> str | None:
    """Return the pass id if ``token`` is authentic, else ``None``.

    Uses a constant-time comparison to avoid signature-timing leaks.
    """
    if not token or token.count(".") < 1:
        return None
    pass_id, _, provided = token.rpartition(".")
    if not pass_id or not provided:
        return None
    expected = sign_pass(pass_id, secret).rpartition(".")[2]
    if hmac.compare_digest(expected, provided):
        return pass_id
    return None
