# Punch Pass "Securify" System — Research

**Author:** research task (branch `claude/punch-pass-securify-research-spa7f8`)
**Date:** 2026-07-20
**Status:** Research / design note

---

## 1. What is being researched

The phrase "punch pass securify system" combines two ideas:

- **Punch pass** — a prepaid, multi-visit pass. A customer buys a bundle of
  visits/services up front (e.g. "10 yoga classes", "5 car washes", "buy 9 get
  the 10th free"). Each use "punches" one slot until the pass is exhausted.
  Modern versions are digital (app, QR code, or Apple/Google Wallet pass)
  rather than a paper card with holes punched in it.
- **"Securify"** — reads as *secure + verify*: making the pass tamper-proof and
  the redemption process fraud-resistant. (Note: **Securify** is also the name
  of an unrelated, now-deprecated static-analysis security scanner for Ethereum
  smart contracts, by ETH Zurich / ChainSecurity. If that tool is what was
  meant, see the appendix.)

This document treats the request as: **how to design and secure a digital
punch-pass system** — the threat model, the attacks it must resist, and a
reference architecture. This is where "punch pass" and "securify" intersect.

---

## 2. How punch passes work

A punch pass is fundamentally a **prepaid credit balance** attached to a
customer, with three lifecycle events:

1. **Issue** — customer purchases a pass with `N` punches for a discounted
   price. This front-loads revenue and locks in repeat visits.
2. **Redeem (punch)** — on each visit the balance is decremented by one. This is
   the security-critical moment.
3. **Expire / exhaust** — the pass ends when the balance hits zero or an
   expiry date passes.

Businesses value them because they reward *frequency* over *value*, are trivial
for customers to understand, and generate visit/spend data. Common users:
cafés, fitness studios, salons, kids' activity classes, car washes.

---

## 3. Threat model — what "securify" has to defend against

A punch pass is money-like: an exhausted or forged pass that still redeems is a
direct loss. The main threats:

| # | Threat | Description |
|---|--------|-------------|
| T1 | **Forgery** | Attacker fabricates a pass that was never purchased. |
| T2 | **Tampering** | Attacker edits a real pass to inflate the balance or extend expiry. |
| T3 | **Double redemption / replay** | Same punch is claimed twice — screenshot of a QR code, cloned pass, offline race. |
| T4 | **Duplication / sharing** | One pass copied and used by many people (unless sharing is intentionally allowed). |
| T5 | **Redemption without presence** | Punches claimed remotely rather than at the point of sale ("farming"). |
| T6 | **Staff/insider fraud** | Employee issues free punches or refunds to themselves. |
| T7 | **Enumeration** | Guessing valid pass IDs to find or hijack passes. |
| T8 | **Repudiation** | Customer or staff disputes that a redemption happened. |

---

## 4. Security controls (the "securify" layer)

### 4.1 Make the pass tamper-proof (defends T1, T2)
- **Cryptographic signing.** The pass is a signed token, not raw editable data.
  Apple Wallet (`.pkpass`) and Google Wallet passes are signed with the
  issuer's certificate; any modification invalidates the signature. A
  self-hosted equivalent is a **JWS/JWT** signed server-side (never trust a
  balance that arrives from the client).
- **Server-authoritative balance.** The device/QR should carry only an
  *identifier* and a signature, never the authoritative punch count. The real
  balance lives in the backend and is the single source of truth. A tampered
  client value simply won't match the server.

### 4.2 Prevent double redemption / replay (defends T3, T5)
- **Unique serialized IDs.** Every pass — and ideally every redemption request —
  carries a unique identifier checked against a registry. A registry reveals
  when a second redemption is attempted for an already-punched item.
- **Atomic decrement.** Redemption must be a single transactional
  read-modify-write (`balance > 0` check and decrement in one DB transaction, or
  an idempotency key) so two concurrent scans can't both succeed.
- **Rotating / one-time codes.** A static QR can be screenshotted and reused.
  Prefer time-based rotating codes (TOTP-style) or per-redemption one-time
  tokens so a captured image expires quickly.
- **Location / presence checks.** Geofencing or a staff-side scan at the counter
  ensures the customer is physically present, blocking remote farming.

### 4.3 Duplication & sharing policy (defends T4)
- Decide explicitly whether a pass is **bearer** (anyone holding it can use it,
  like a paper card) or **bound** to an account/identity. Bound passes require
  login or a linked phone number at redemption; bearer passes accept the
  double-redemption risk in exchange for convenience.

### 4.4 Access control & auditing (defends T6, T8)
- **Staff authentication** on the redemption device; every punch logged with
  who, when, where.
- **Immutable audit log** of issue/redeem/refund events for dispute resolution
  (non-repudiation). Blockchain/ledger-backed variants exist for this but a
  well-secured append-only DB log is sufficient for most businesses.

### 4.5 Identifier hygiene (defends T7)
- Use non-sequential, high-entropy pass IDs (UUIDv4 or random 128-bit) so IDs
  can't be guessed or enumerated.

---

## 5. Reference architecture

```
 ┌─────────────┐    buy pass     ┌──────────────────┐
 │  Customer    │ ─────────────▶ │  Backend / API    │
 │  wallet/app  │ ◀───────────── │  (source of truth)│
 └─────────────┘  signed pass    └────────┬─────────┘
        │  (ID + signature)                │
        │                                  │  ┌───────────────┐
        │  show QR at counter              ├─▶│ Passes table   │ balance, expiry
        ▼                                  │  │ (per pass)     │
 ┌─────────────┐   scan + verify   ┌───────┴──┐└───────────────┘
 │ Staff device │ ───────────────▶ │ Redeem   │  ┌───────────────┐
 │ (POS/kiosk)  │ ◀─────────────── │ engine   ├─▶│ Redemption log │ append-only
 └─────────────┘   approve/deny    └──────────┘  │ (audit)        │
                                                  └───────────────┘
```

**Redemption flow (the critical path):**
1. Staff device scans the customer's QR/NFC → extracts pass ID + signature.
2. Backend verifies the signature (rejects forged/tampered — T1/T2).
3. Backend checks the code hasn't already been used and isn't expired
   (rejects replay — T3), optionally checks geolocation (T5).
4. Backend **atomically** verifies `balance > 0` and decrements
   (prevents concurrent double-spend — T3).
5. Backend appends an immutable audit entry (T6/T8) and returns approve/deny.
6. Client display is updated *from the server response*, never trusted on its own.

**Build vs. buy:** Apple/Google Wallet (via providers like PassKit) give you
signed, tamper-proof passes and push updates out of the box, which covers T1/T2
cheaply. You still own the redemption engine, registry, atomic decrement, and
audit log — that is where most of the real "securify" work lives.

---

## 6. Recommendations (priority order)

1. **Server-authoritative balance + atomic decrement.** Non-negotiable; defeats
   the highest-impact attacks (tamper + double-spend) with the least effort.
2. **Signed passes** (Wallet certs or server-side JWS). Cheap forgery/tamper
   protection.
3. **One-time or rotating redemption codes** instead of static QR images.
4. **Unique high-entropy IDs + a redemption registry** for replay detection.
5. **Immutable audit log + staff auth** for insider fraud and disputes.
6. **Explicit bearer-vs-bound policy** and optional geofencing, chosen to match
   the business's fraud tolerance vs. friction budget.

---

## Appendix — the *other* "Securify"

If "Securify" referred to the **smart-contract scanner** rather than "secure +
verify": Securify is a free, open-source static-analysis security scanner for
Ethereum smart contracts, built by ICE Center / ETH Zurich and ChainSecurity.
Unlike symbolic-execution tools (Oyente, Mythril), it statically analyzes *all*
paths of a contract. **Securify v2.0** analyzes Solidity source (not just EVM
bytecode), uses a declarative Datalog analysis on the Soufflé engine, and covers
37 vulnerability classes from the Smart Contract Weakness Classification (SWC)
registry. Both the original and v2 repositories are now marked **deprecated** on
GitHub. If the intended task is actually to run a security scan of smart-contract
code (or of the app in this repo), say so and the research can be re-scoped.

---

## Sources

- ChainSecurity — Securify on GitHub: https://medium.com/chainsecurity/securify-is-now-on-github-d3bec281eafc
- Securify v2.0 release notes: https://medium.com/chainsecurity/release-of-securify-v2-0-6304a40034f
- eth-sri/securify2 (GitHub): https://github.com/eth-sri/securify2
- iClassPro — What Are Punch Passes: https://support.iclasspro.com/hc/en-us/articles/360043355553-What-Are-Punch-Passes
- Voucherify — Punch loyalty programs: https://www.voucherify.io/blog/why-and-how-to-run-a-punch-loyalty-program
- Coupontools — Digital punch card solutions: https://www.coupontools.com/en/blog/332/digital-punch-card-loyalty-solutions-for-modern-businesses
- MDPI Sensors — Tamper-Proof QR Code Generation and Fraud-Resistant Verification: https://www.mdpi.com/1424-8220/25/13/3855
- PassKit — Introduction to Apple & Google Wallet Passes: https://help.passkit.com/en/articles/6608981-introduction-to-apple-google-wallet-passes
- Photon — Tokenization of loyalty systems: https://www.photon.com/tokenization-supercharging-the-traditional-loyalty-system
