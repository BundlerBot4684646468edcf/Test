"""Secure digital punch-pass system.

Implements the security controls described in
``docs/punch-pass-securify-research.md``:

* server-authoritative balance with atomic decrement (T3)
* HMAC-signed, tamper-proof pass tokens (T1, T2)
* one-time redemption codes to defeat replay/screenshot reuse (T3)
* high-entropy, non-enumerable pass ids (T7)
* an append-only audit log for non-repudiation (T6, T8)
* staff API-key authentication on redemption endpoints (T6)
"""

from .service import PunchPassService, RedemptionError

__all__ = ["PunchPassService", "RedemptionError"]
