"""Tests exercising the security controls end-to-end.

Each test maps to a threat from docs/punch-pass-securify-research.md.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punchpass import PunchPassService, security  # noqa: E402


@pytest.fixture()
def svc(tmp_path):
    return PunchPassService(db_path=str(tmp_path / "t.db"), secret="unit-test-secret")


# ---- signing / tamper resistance (T1, T2) -------------------------------
def test_token_roundtrip(svc):
    p, token = svc.issue_pass("alice", 5)
    assert svc.resolve_token(token).id == p.id


def test_tampered_token_rejected(svc):
    _, token = svc.issue_pass("alice", 5)
    pass_id, _, sig = token.rpartition(".")
    forged = pass_id[:-1] + ("x" if pass_id[-1] != "x" else "y") + "." + sig
    assert svc.resolve_token(forged) is None


def test_forged_signature_rejected(svc):
    p, _ = svc.issue_pass("alice", 5)
    assert security.verify_pass(f"{p.id}.deadbeef", "unit-test-secret") is None


# ---- balance is server-authoritative + atomic decrement (T3) ------------
def test_redeem_decrements(svc):
    _, token = svc.issue_pass("bob", 3)
    code = svc.issue_redemption_code(token)
    p = svc.redeem(code, staff="s1")
    assert p.balance == 2


def test_cannot_overdraw(svc):
    _, token = svc.issue_pass("bob", 1)
    svc.redeem(svc.issue_redemption_code(token), staff="s1")
    with pytest.raises(Exception) as ei:
        svc.redeem(svc.issue_redemption_code(token), staff="s1")
    assert ei.value.reason == "no_balance"


# ---- one-time redemption codes / replay (T3) ----------------------------
def test_code_is_single_use(svc):
    _, token = svc.issue_pass("carol", 5)
    code = svc.issue_redemption_code(token)
    svc.redeem(code, staff="s1")
    with pytest.raises(Exception) as ei:
        svc.redeem(code, staff="s1")  # replayed screenshot
    assert ei.value.reason == "invalid_code"


def test_expired_code_rejected(svc):
    _, token = svc.issue_pass("carol", 5)
    code = svc.issue_redemption_code(token, ttl_seconds=5)
    # force expiry
    svc._conn.execute(
        "UPDATE redemption_codes SET expires_at = '2000-01-01T00:00:00+00:00' WHERE code = ?",
        (code,),
    )
    with pytest.raises(Exception) as ei:
        svc.redeem(code, staff="s1")
    assert ei.value.reason == "code_expired"


# ---- non-enumerable ids (T7) --------------------------------------------
def test_ids_are_random(svc):
    ids = {svc.issue_pass("u", 1)[0].id for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("pp_") for i in ids)


# ---- audit trail (T6, T8) -----------------------------------------------
def test_audit_records_issue_and_redeem(svc):
    p, token = svc.issue_pass("dave", 2)
    svc.redeem(svc.issue_redemption_code(token), staff="counter-1")
    events = [e["event"] for e in svc.audit_trail(p.id)]
    assert events == ["issue", "redeem"]


def test_denied_redeem_is_audited(svc):
    p, token = svc.issue_pass("dave", 1)
    svc.redeem(svc.issue_redemption_code(token), staff="counter-1")
    try:
        svc.redeem(svc.issue_redemption_code(token), staff="counter-1")
    except Exception:
        pass
    events = [e["event"] for e in svc.audit_trail(p.id)]
    assert "redeem_denied" in events


# ---- HTTP layer + staff auth (T6) ---------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PUNCHPASS_DB", str(tmp_path / "api.db"))
    monkeypatch.setenv("PUNCHPASS_SECRET", "api-secret")
    monkeypatch.setenv("PUNCHPASS_STAFF_KEY", "topsecret")
    # import fresh so env vars are picked up
    for m in list(sys.modules):
        if m.startswith("punchpass"):
            del sys.modules[m]
    from punchpass import api
    return TestClient(api.app), api


def test_issue_requires_staff_key(client):
    c, _ = client
    r = c.post("/passes", json={"customer": "e", "punches": 3})
    assert r.status_code == 401


def test_full_http_flow(client):
    c, _ = client
    hdr = {"x-api-key": "topsecret"}
    r = c.post("/passes", json={"customer": "e", "punches": 2}, headers=hdr)
    assert r.status_code == 200
    token = r.json()["token"]

    # customer mints a rotating code (no staff key needed)
    code = c.post("/redemption-code", json={"token": token}).json()["code"]

    # staff redeems
    r = c.post("/redeem", json={"code": code}, headers=hdr)
    assert r.status_code == 200 and r.json()["balance"] == 1

    # replay the same code -> rejected
    r = c.post("/redeem", json={"code": code}, headers=hdr)
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "invalid_code"


def test_view_by_token_never_trusts_client(client):
    c, _ = client
    hdr = {"x-api-key": "topsecret"}
    token = c.post("/passes", json={"customer": "e", "punches": 4}, headers=hdr).json()["token"]
    r = c.get(f"/passes/by-token/{token}")
    assert r.status_code == 200 and r.json()["balance"] == 4
