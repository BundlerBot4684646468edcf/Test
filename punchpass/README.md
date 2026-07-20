# Punch Pass — Securify

A secure digital **punch-pass** system: prepaid multi-visit passes (e.g. "10
classes", "buy 9 get the 10th free") with the anti-fraud controls described in
[`../docs/punch-pass-securify-research.md`](../docs/punch-pass-securify-research.md).

## What it implements

| Control | Where | Threat defended |
|---------|-------|-----------------|
| HMAC-signed, tamper-proof pass tokens | `security.py` | forgery / tampering (T1, T2) |
| Server-authoritative balance | `service.py`, `db.py` | tampering (T2) |
| Atomic, guarded decrement (`balance > 0`) in one transaction | `service.redeem` | double-spend (T3) |
| One-time, short-lived redemption codes | `service.issue_redemption_code` | replay / screenshot reuse (T3) |
| Per-thread connections + WAL + busy_timeout | `service.py`, `db.py` | concurrent-writer correctness |
| High-entropy, non-sequential ids | `security.new_id` | enumeration (T7) |
| Append-only audit log (issue / redeem / redeem_denied) | `db.audit_log` | insider fraud, repudiation (T6, T8) |
| Staff API-key auth on privileged endpoints | `api.require_staff` | insider fraud (T6) |

The customer's QR/wallet carries only a **signed id** — never the balance. The
real balance lives server-side and is the single source of truth.

## Run it

```bash
pip install -r ../requirements.txt

export PUNCHPASS_SECRET="a-long-random-string"     # signs pass tokens
export PUNCHPASS_STAFF_KEY="another-random-string" # staff redemption auth
export PUNCHPASS_DB="punchpass.db"

uvicorn punchpass.api:app --port 8099
```

## API

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/passes` | staff | issue a pass, returns signed token |
| `POST` | `/redemption-code` | none | customer mints a one-time punch code from a token |
| `POST` | `/redeem` | staff | consume one punch using a code |
| `GET`  | `/passes/by-token/{token}` | none | read-only balance from a signed token |
| `GET`  | `/passes/{id}/audit` | staff | audit trail for a pass |
| `GET`  | `/health` | none | liveness |

### Typical flow

```
staff  POST /passes            -> { token, balance: 10 }        # issue
customer POST /redemption-code -> { code }                      # rotating one-time code
staff  POST /redeem {code}     -> { balance: 9 }                # punch
staff  POST /redeem {code}     -> 409 invalid_code              # replay blocked
```

## Test

```bash
python -m pytest tests/test_punchpass.py -q
```

The suite covers tamper rejection, overdraw prevention, single-use codes,
expiry, non-repudiation logging, and the HTTP flow including staff auth.

## Notes / production hardening

- Set strong `PUNCHPASS_SECRET` / `PUNCHPASS_STAFF_KEY`; the defaults are for
  local dev only.
- Add per-staff keys (not a shared key) and rate-limiting for real deployments.
- Optional controls from the research note not built here: geofencing / presence
  checks (T5), Apple/Google Wallet pass issuance, and account-bound redemption
  (`bound` is stored on the pass but not yet enforced at redemption).
