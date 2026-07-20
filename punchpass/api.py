"""FastAPI HTTP layer for the punch-pass system.

Endpoints are split by trust level:

* customer-facing: mint a rotating redemption code from a signed token
* staff-only (API-key protected): issue passes, redeem, read audit trail
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .service import PunchPassService, RedemptionError

DB_PATH = os.getenv("PUNCHPASS_DB", "punchpass.db")
SECRET = os.getenv("PUNCHPASS_SECRET", "change-me")
STAFF_API_KEY = os.getenv("PUNCHPASS_STAFF_KEY", "dev-staff-key")

app = FastAPI(title="Punch Pass — Securify", version="1.0.0")
service = PunchPassService(db_path=DB_PATH, secret=SECRET)


def require_staff(x_api_key: str = Header(default="")) -> str:
    """Authenticate staff for privileged endpoints (T6)."""
    import hmac

    if not x_api_key or not hmac.compare_digest(x_api_key, STAFF_API_KEY):
        raise HTTPException(status_code=401, detail="invalid staff API key")
    return "staff:" + x_api_key[:4]


# ---- schemas ------------------------------------------------------------
class IssueRequest(BaseModel):
    customer: str
    punches: int = Field(gt=0, le=1000)
    bound: bool = False
    valid_days: int | None = 365


class IssueResponse(BaseModel):
    pass_id: str
    token: str
    balance: int
    total_punches: int


class CodeRequest(BaseModel):
    token: str
    ttl_seconds: int = Field(default=60, ge=5, le=600)


class RedeemRequest(BaseModel):
    code: str


class PassView(BaseModel):
    pass_id: str
    customer: str
    balance: int
    total_punches: int
    bound: bool
    expires_at: str | None


# ---- staff endpoints ----------------------------------------------------
@app.post("/passes", response_model=IssueResponse)
def issue(req: IssueRequest, staff: str = Depends(require_staff)):
    p, token = service.issue_pass(
        req.customer, req.punches, bound=req.bound, valid_days=req.valid_days, actor=staff
    )
    return IssueResponse(pass_id=p.id, token=token, balance=p.balance, total_punches=p.total_punches)


@app.post("/redeem", response_model=PassView)
def redeem(req: RedeemRequest, staff: str = Depends(require_staff)):
    try:
        p = service.redeem(req.code, staff=staff)
    except RedemptionError as e:
        raise HTTPException(status_code=409, detail={"reason": e.reason, "message": e.message})
    return PassView(
        pass_id=p.id, customer=p.customer, balance=p.balance,
        total_punches=p.total_punches, bound=p.bound, expires_at=p.expires_at,
    )


@app.get("/passes/{pass_id}/audit")
def audit(pass_id: str, staff: str = Depends(require_staff)):
    if service.get_pass(pass_id) is None:
        raise HTTPException(status_code=404, detail="pass not found")
    return {"pass_id": pass_id, "events": service.audit_trail(pass_id)}


# ---- customer endpoints -------------------------------------------------
@app.post("/redemption-code")
def redemption_code(req: CodeRequest):
    """Customer's app exchanges its signed token for a short-lived punch code."""
    try:
        code = service.issue_redemption_code(req.token, ttl_seconds=req.ttl_seconds)
    except RedemptionError as e:
        raise HTTPException(status_code=400, detail={"reason": e.reason, "message": e.message})
    return {"code": code, "ttl_seconds": req.ttl_seconds}


@app.get("/passes/by-token/{token}", response_model=PassView)
def view_by_token(token: str):
    """Read-only balance lookup from a signed token (never trusts client state)."""
    p = service.resolve_token(token)
    if p is None:
        raise HTTPException(status_code=404, detail="invalid or unknown pass token")
    return PassView(
        pass_id=p.id, customer=p.customer, balance=p.balance,
        total_punches=p.total_punches, bound=p.bound, expires_at=p.expires_at,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
