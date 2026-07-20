"""Business logic for the punch-pass system.

Everything security-critical lives here, independent of the HTTP layer so it can
be unit-tested directly.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import db, security


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


class RedemptionError(Exception):
    """Raised when a redemption is refused. ``reason`` is a stable code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass
class Pass:
    id: str
    customer: str
    total_punches: int
    balance: int
    bound: bool
    created_at: str
    expires_at: str | None

    @classmethod
    def from_row(cls, row) -> "Pass":
        return cls(
            id=row["id"],
            customer=row["customer"],
            total_punches=row["total_punches"],
            balance=row["balance"],
            bound=bool(row["bound"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )


class PunchPassService:
    def __init__(self, db_path: str = "punchpass.db", secret: str = "change-me"):
        if not secret or secret == "change-me":
            # A weak signing secret undermines T1/T2; refuse to run silently.
            import os

            secret = os.getenv("PUNCHPASS_SECRET", secret)
        self._db_path = db_path
        self._secret = secret
        # A sqlite3 connection cannot run concurrent transactions, so give each
        # thread its own connection to the same WAL database. Writers then queue
        # on the file lock (busy_timeout) instead of colliding on one connection.
        self._local = threading.local()
        db.connect(db_path).close()  # ensure schema exists up front

    @property
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = db.connect(self._db_path)
            self._local.conn = conn
        return conn

    # ---- audit -----------------------------------------------------------
    def _audit(self, conn, event: str, pass_id: str | None, actor: str | None, detail: str = ""):
        conn.execute(
            "INSERT INTO audit_log (ts, event, pass_id, actor, detail) VALUES (?,?,?,?,?)",
            (_iso(_now()), event, pass_id, actor, detail),
        )

    # ---- issue -----------------------------------------------------------
    def issue_pass(
        self,
        customer: str,
        punches: int,
        *,
        bound: bool = False,
        valid_days: int | None = 365,
        actor: str = "system",
    ) -> tuple[Pass, str]:
        """Create a pass and return (pass, signed_token).

        The token is what the customer stores; it carries only the signed id.
        """
        if punches <= 0:
            raise ValueError("punches must be positive")
        pass_id = security.new_id(prefix="pp_")
        created = _now()
        expires = created + timedelta(days=valid_days) if valid_days else None
        with db.transaction(self._conn) as conn:
            conn.execute(
                "INSERT INTO passes (id, customer, total_punches, balance, bound, created_at, expires_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (pass_id, customer, punches, punches, int(bound), _iso(created), _iso(expires)),
            )
            self._audit(conn, "issue", pass_id, actor, f"customer={customer} punches={punches}")
        token = security.sign_pass(pass_id, self._secret)
        return self.get_pass(pass_id), token

    # ---- read ------------------------------------------------------------
    def get_pass(self, pass_id: str) -> Pass | None:
        row = self._conn.execute("SELECT * FROM passes WHERE id = ?", (pass_id,)).fetchone()
        return Pass.from_row(row) if row else None

    def resolve_token(self, token: str) -> Pass | None:
        """Verify a signed token and return the current server-side state."""
        pass_id = security.verify_pass(token, self._secret)
        return self.get_pass(pass_id) if pass_id else None

    # ---- redemption codes ------------------------------------------------
    def issue_redemption_code(self, token: str, ttl_seconds: int = 60) -> str:
        """Mint a short-lived one-time code for the pass in ``token``.

        Rotating codes mean a screenshot of yesterday's QR is worthless (T3).
        """
        pass_id = security.verify_pass(token, self._secret)
        if not pass_id or not self.get_pass(pass_id):
            raise RedemptionError("invalid_token", "Pass token is invalid.")
        code = security.new_redemption_code()
        now = _now()
        with db.transaction(self._conn) as conn:
            conn.execute(
                "INSERT INTO redemption_codes (code, pass_id, created_at, expires_at) VALUES (?,?,?,?)",
                (code, pass_id, _iso(now), _iso(now + timedelta(seconds=ttl_seconds))),
            )
        return code

    # ---- redeem ----------------------------------------------------------
    def redeem(self, code: str, *, staff: str) -> Pass:
        """Consume one punch using a one-time code. Atomic and single-use.

        Raises ``RedemptionError`` on any failure. Both the state mutation and
        the audit record are committed regardless of the allow/deny outcome, so
        refused attempts (replay, overdraw, expiry) are never lost from the log.
        """
        now = _now()
        # Consume the code, decrement, and audit all in ONE transaction so the
        # punch and its log entry are atomic — no window where a punch is spent
        # but unlogged, or logged but not spent. We never raise from inside the
        # block (that would roll back the code-consumption and the denial audit);
        # instead we record the outcome and raise only after the commit.
        _MESSAGES = {
            "invalid_code": "Redemption code is invalid or already used.",
            "code_expired": "Redemption code has expired.",
            "pass_missing": "Pass no longer exists.",
            "pass_expired": "Pass has expired.",
            "no_balance": "Pass has no punches remaining.",
        }
        pass_id: str | None = None
        reason: str | None = None
        with db.transaction(self._conn) as conn:
            crow = conn.execute(
                "SELECT * FROM redemption_codes WHERE code = ?", (code,)
            ).fetchone()
            if crow is None:
                # Unknown or already-consumed code -> replay / forgery attempt.
                reason = "invalid_code"
            else:
                pass_id = crow["pass_id"]
                # One-time: delete the code so it can never be reused.
                conn.execute("DELETE FROM redemption_codes WHERE code = ?", (code,))
                prow = conn.execute("SELECT * FROM passes WHERE id = ?", (pass_id,)).fetchone()
                if crow["expires_at"] < _iso(now):
                    reason = "code_expired"
                elif prow is None:
                    reason = "pass_missing"
                elif prow["expires_at"] is not None and prow["expires_at"] < _iso(now):
                    reason = "pass_expired"
                else:
                    # Atomic, guarded decrement: the WHERE clause makes concurrent
                    # double-spend impossible — only one writer moves balance 1 -> 0.
                    cur = conn.execute(
                        "UPDATE passes SET balance = balance - 1 WHERE id = ? AND balance > 0",
                        (pass_id,),
                    )
                    if cur.rowcount != 1:
                        reason = "no_balance"

            if reason is None:
                self._audit(conn, "redeem", pass_id, staff, "punch")
                prow = conn.execute("SELECT * FROM passes WHERE id = ?", (pass_id,)).fetchone()
                result = Pass.from_row(prow)
            else:
                self._audit(conn, "redeem_denied", pass_id, staff, reason)
                result = None

        if reason is not None:
            raise RedemptionError(reason, _MESSAGES[reason])
        return result

    # ---- audit read ------------------------------------------------------
    def audit_trail(self, pass_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, event, actor, detail FROM audit_log WHERE pass_id = ? ORDER BY id",
            (pass_id,),
        ).fetchall()
        return [dict(r) for r in rows]
