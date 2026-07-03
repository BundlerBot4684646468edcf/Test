"""Per-salon dashboard login.

Each salon gets an admin password (set or auto-generated at onboarding).
A successful login sets an HMAC-signed session cookie scoped to
/salons/{slug}, so one salon's session never opens another salon's
dashboard. No external dependencies — PBKDF2 for password hashes, HMAC
for session tokens.

Secrets:
- SESSION_SECRET (env): signs session cookies. If unset, a random secret
  is generated per process start — fine for dev, but sessions then die on
  every restart, so set it in production.
"""

import base64
import hashlib
import hmac
import os
import secrets
import time

COOKIE_NAME = "salon_session"
SESSION_TTL_SECONDS = 12 * 3600

_RUNTIME_SECRET = secrets.token_hex(32)


def _secret() -> bytes:
    return os.getenv("SESSION_SECRET", _RUNTIME_SECRET).encode()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return hmac.compare_digest(check.hex(), digest)


def make_session_token(salon_slug: str) -> str:
    payload = f"{salon_slug}|{int(time.time()) + SESSION_TTL_SECONDS}"
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(payload.encode()).decode() + "." + sig


def verify_session_token(token: str | None, salon_slug: str) -> bool:
    if not token or "." not in token:
        return False
    encoded, sig = token.rsplit(".", 1)
    try:
        payload = base64.urlsafe_b64decode(encoded.encode()).decode()
    except Exception:
        return False
    expected = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False
    slug, _, expires = payload.partition("|")
    return slug == salon_slug and expires.isdigit() and int(expires) > time.time()


def render_login_page(salon_name: str, salon_slug: str) -> str:
    return LOGIN_HTML.replace("__SALON_NAME__", salon_name).replace("__SALON_SLUG__", salon_slug)


LOGIN_HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SALON_NAME__ — Login</title>
<style>
  :root { --line: #e4e2dd; --bg: #faf9f7; --ink: #2b2a27; --muted: #8a867e; --accent: #4a6b5d; --danger: #b3552f; }
  * { box-sizing: border-box; margin: 0; }
  body {
    font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--ink);
    min-height: 100vh; display: grid; place-items: center; padding: 24px;
  }
  form {
    background: #fff; border: 1px solid var(--line); border-radius: 12px;
    padding: 28px; width: 100%; max-width: 360px; display: grid; gap: 14px;
  }
  h1 { font-size: 17px; font-weight: 600; }
  h1 small { display: block; color: var(--muted); font-weight: 400; font-size: 13px; margin-top: 4px; }
  input {
    border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
    font-size: 15px; font-family: inherit; width: 100%;
  }
  button {
    background: var(--accent); border: none; color: #fff; border-radius: 8px;
    padding: 10px 12px; font-size: 15px; cursor: pointer;
  }
  button:hover { opacity: 0.9; }
  #err { color: var(--danger); font-size: 13px; min-height: 18px; }
</style>
</head>
<body>
<form id="f">
  <h1>__SALON_NAME__ <small>Anmeldung zur Verwaltung</small></h1>
  <input type="password" id="pw" placeholder="Passwort" autofocus autocomplete="current-password">
  <div id="err"></div>
  <button type="submit">Anmelden</button>
</form>
<script>
document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("err");
  err.textContent = "";
  const resp = await fetch("/salons/__SALON_SLUG__/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password: document.getElementById("pw").value }),
  });
  if (resp.ok) {
    const next = new URLSearchParams(location.search).get("next");
    location.href = next && next.startsWith("/salons/__SALON_SLUG__/") ? next : "/salons/__SALON_SLUG__/admin";
  } else {
    err.textContent = (await resp.json().catch(() => ({}))).detail || "Anmeldung fehlgeschlagen";
  }
});
</script>
</body>
</html>
"""
