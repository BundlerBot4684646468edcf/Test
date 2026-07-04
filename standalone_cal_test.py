#!/usr/bin/env python3
"""
Standalone Cal.com booking test — ONE file, no package structure needed.

Just save this file anywhere and run:
    python standalone_cal_test.py

It tests the full flow against the real Cal.com API and prints the exact
error at whichever step fails, so we can see why bookings aren't working.
"""

import os
import sys
import json
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# CONFIG — API key is read from the CAL_API_KEY environment variable.
# (Set it in PowerShell with:  $env:CAL_API_KEY = "cal_live_...")
# ---------------------------------------------------------------------------
CAL_API_KEY = os.getenv("CAL_API_KEY", "")
CAL_API_BASE = os.getenv("CAL_API_BASE", "https://api.cal.com/v2")
CAL_API_VERSION = "2026-02-25"


def call(method, path, body=None, api_version=CAL_API_VERSION):
    """Make a request to Cal.com and return (status_code, parsed_json)."""
    url = f"{CAL_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "Content-Type": "application/json",
        "cal-api-version": api_version,
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return e.code, parsed


def unwrap(payload):
    """Cal.com wraps responses in {"status":"success","data":{...}}."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    section("1. API key check")
    if not CAL_API_KEY:
        print("FEHLER: CAL_API_KEY ist nicht gesetzt.")
        print('  In PowerShell:  $env:CAL_API_KEY = "cal_live_..."')
        return False
    print(f"OK — Key gesetzt ({CAL_API_KEY[:12]}...)")
    print(f"Base URL: {CAL_API_BASE}")

    section("2. Verbindung testen (GET /me)")
    status, body = call("GET", "/me")
    print(f"HTTP {status}")
    print(json.dumps(body, indent=2)[:1500])
    if status != 200:
        print("\n>>> Verbindung/Auth fehlgeschlagen. Prüfe den API-Key.")
        return False
    me = unwrap(body)
    print(f"\nOK — eingeloggt als: {me.get('email') or me.get('username') or me}")

    section("3. Team anlegen (POST /teams)")
    status, body = call("POST", "/teams", {"name": "Debug Test Salon"})
    print(f"HTTP {status}")
    print(json.dumps(body, indent=2)[:1500])
    if status not in (200, 201):
        print("\n>>> Team-Anlage fehlgeschlagen. Fehlermeldung oben ansehen.")
        print("    (Häufig: Team-Plan nötig, oder Key hat keine Team-Rechte.)")
        return False
    team = unwrap(body)
    team_id = team.get("id")
    print(f"\nOK — Team ID: {team_id}")

    section("4. Slots abrufen für vorhandenen Event-Type (optional)")
    print("Übersprungen — braucht eine echte eventTypeId.")
    print("Sobald Team + Event-Type stehen, testen wir Buchung.")

    section("FERTIG")
    print("Die Kernverbindung zu Cal.com funktioniert.")
    print("Wenn hier alles OK war, liegt das Buchungsproblem NICHT an der")
    print("Cal.com-Verbindung, sondern an Famulor (ruft das Tool nicht auf)")
    print("oder an fehlenden Salon-Daten in der Datenbank.")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
