"""SQLite storage layer.

The database is the single source of truth for every pass balance. SQLite's
transactional guarantees give us the atomic decrement that prevents two
concurrent scans from both succeeding (T3).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS passes (
    id            TEXT PRIMARY KEY,
    customer      TEXT NOT NULL,
    total_punches INTEGER NOT NULL,
    balance       INTEGER NOT NULL,
    bound         INTEGER NOT NULL DEFAULT 0,   -- 1 = account-bound, 0 = bearer
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    CHECK (balance >= 0),
    CHECK (balance <= total_punches)
);

-- Outstanding one-time redemption codes. A code is deleted the instant it is
-- consumed, so a replayed / screenshotted code finds nothing to consume (T3).
CREATE TABLE IF NOT EXISTS redemption_codes (
    code       TEXT PRIMARY KEY,
    pass_id    TEXT NOT NULL REFERENCES passes(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

-- Append-only audit trail for non-repudiation and insider-fraud review (T6, T8).
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,          -- issue | redeem | redeem_denied
    pass_id    TEXT,
    actor      TEXT,                   -- staff id / device
    detail     TEXT
);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # Make concurrent writers wait for the lock instead of failing immediately.
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside an IMMEDIATE transaction (serialises writers)."""
    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    else:
        conn.execute("COMMIT;")
