# Master TODO — AI Receptionist for Salons

Living checklist. Ordered by urgency. Detail lives in OFFER.md, PILOT-ANDRE.md,
AGENT-ANDRE.md, BREAK-TEST-20.md.

## 🔴 NOW — before Andre's virtual demo (this week)

- [ ] Fix Famulor prompt: (1) refuse Monday/Sunday, (2) capture + repeat back
      mobile number, (3) check calendar before confirming a slot
- [ ] Wire post-call confirmation: number → webhook → ONE message, caller's language
- [ ] Set up the calendar (Cal.com free or Famulor native) with services,
      durations, and Andre's hours as availability; seed a few fake bookings
- [ ] Run all 20 break tests yourself; log PASS/FAIL; fix fails; re-run
- [ ] Confirm demo-ready bar: 1–10 all PASS, no hallucinated bookings 14–20
- [ ] Onboarding call with Andre: real service list + prices + durations,
      never-book list, which number actually rings in the salon
- [ ] Send Andre the web-call link/virtual number + the 10-scenario test script

## 🟠 GO-LIVE — turn the demo into the 30-day pilot

- [ ] **Start the +39 number KYC application NOW** (days–2 weeks; the long pole)
- [ ] Decide install: sequential-ring on our number vs `**61*` forwarding on his;
      confirm on his actual phone which works
- [ ] WhatsApp Business API set up (win messages + confirmations); until then,
      a temporary SMS/manual fallback
- [ ] Test the FULL chain end to end: call → book → confirmation → 24h reminder
      → review request (fires ONCE)
- [ ] One-page pilot agreement (DE): 30 days free, ≥10-bookings guarantee,
      testimonial + 2 referrals in exchange, then 149 €/mo — get it signed
- [ ] Forward closed-day / after-hours calls too (Mo/So/after 19:00 = found money)
- [ ] In-salon go-live moment: Andre calls his own number, it books, his phone buzzes

## 🟡 LEGAL / COMPANY / MONEY (the deal-killers if skipped)

- [ ] Company entity + can you legally invoice in IT/DE? VAT/USt handling
- [ ] How Andre pays after day 30: SEPA direct debit / Stripe subscription — set up
- [ ] AVV / DPA (Auftragsverarbeitungsvertrag) template — every salon signs one
- [ ] Call-recording consent handling (GDPR) + data stored in EU + deletion policy
- [ ] Impressum + Datenschutzerklärung for the salon website before it publishes
- [ ] AI-disclosure confirmed in the agent's first sentence (EU AI Act) ✅ in prompt
- [ ] Decide number-ownership policy (we own it = lock-in; be transparent about it)

## 🟢 MEASUREMENT — so day 30 has a real report

- [ ] Define how you capture: calls handled, bookings made, € recovered,
      Mo/So/after-hours bookings, fallback rate, hang-up rate
- [ ] Where the data lives (Famulor export? sheet? small DB?)
- [ ] Track OUR cost per salon (telephony + LLM + WhatsApp) = margin baseline
- [ ] Build the day-30 one-page report template (euros first)
- [ ] Weekly ritual: listen to the 3 worst calls, fix prompt/data

## 🔵 AFTER ANDRE CONVERTS — "implement everything" (in priority order)

- [ ] Rebooking already live in prompt ✅ — confirm it's actually firing
- [ ] Publish the website (real prices, verbatim reviews, logo/photos, Impressum)
- [ ] No-show reminders live (24h before, confirm/reschedule)
- [ ] Review-request automation (grow the 4.8★ base)
- [ ] Weekly digest replaces per-booking pings
- [ ] Fill dead slots / waitlist backfill on cancellations

## ⚪ SCALE PREP — after 1 paying salon (don't do early)

- [ ] Turn onboarding into a repeatable SOP / checklist (target < 1h)
- [ ] Case-study one-pager from Andre's 60-day numbers
- [ ] Line up prospects 2–5 (use Andre's numbers + 2 referral intros)
- [ ] Templatize: new salon = clone agent + swap data + new number
- [ ] Decide expansion order (barbers/nails adjacent; dentists = later, diff compliance)

## Open questions to resolve

- [ ] Which number rings in Andre's salon — 0473 landline or 339 mobile?
- [ ] Does Famulor connect to Cal.com natively, or do we go via webhook/Make?
- [ ] SMS vs WhatsApp as the primary customer channel for confirmations?
- [ ] Italian-geographic +39 number vs national/mobile-format (KYC difficulty)?
