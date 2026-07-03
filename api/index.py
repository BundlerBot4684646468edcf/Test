"""Vercel serverless entrypoint: exposes the FastAPI app as an ASGI handler.

Vercel's Python runtime detects the module-level `app` variable and routes
all requests here via the rewrite in vercel.json. On Vercel, set
SALON_DATABASE_URL to a Postgres URL (Neon / Vercel Postgres) — the
serverless filesystem is ephemeral, so SQLite would lose all bookings."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salon.main import app  # noqa: E402,F401
