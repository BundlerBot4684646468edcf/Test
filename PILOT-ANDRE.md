# Pilot #1 — hair.style.by andré (Meran)

Status: **committed verbally** — saw the preview site, values the AI receptionist
highest. Wants to TEST first, then implement everything.

## Customer facts

| | |
|---|---|
| Salon | hair.style.by andré, Via Monte Tessa 16/B, 39012 Meran (BZ) |
| Phone (salon) | +39 0473 445377 — **the only booking channel** |
| Mobile/WhatsApp | +39 339 585 8111 |
| Hours | Di–Fr 8–19, Sa 8–16, **Mo + So geschlossen** |
| Booking system | none (Fresha page is a listing only, no online booking) |
| Reputation | 4.8★ Google, praised for spontaneous appointments, modern cuts/colors, rockabilly flair |
| Web | Facebook (andrehair80) + our preview site (unpublished) |

**Segment: "Buchungssystem: nichts" → we ARE the system. No integration work.
Highest lock-in.**

## Deal terms (put in writing, one page, before go-live)

- 30 days free test of the AI receptionist
- Guarantee framing: if it books fewer than 10 appointments in the month, it stays free
- In exchange: testimonial + case-study numbers + 2 salon-owner intros
- After the test: full package at pilot price **149 €/Monat lifetime** (list is 199),
  setup fee waived as pilot perk
- He can switch the forwarding off himself at any time (`##61#`) — say this out loud

## Test phase (Days 1–30): AI receptionist ONLY

Deliberately narrow. He tests ONE thing: "does the phone stop leaking bookings?"
Website publishing, review automation, no-show reminders all wait — they are the
"implement everything" reward after the test convinces him.

### Go-live checklist

**Onboarding call (30 min) — collect:**
- [ ] Real service list + prices + durations (site prices are placeholders)
- [ ] Who is bookable (just André? anyone else?)
- [ ] Slot granularity + buffer preferences (e.g. cut 30 min, color 2 h)
- [ ] How he wants walk-in/„spontan" requests handled (his trademark —
      the agent should offer same-day slots if the calendar allows)
- [ ] WhatsApp number for win messages (339 585 8111)
- [ ] Consent: call recording/transcripts (GDPR), AVV signed
- [ ] What the agent must NOT do (e.g. never book color corrections — callback instead)

**Tech setup:**
- [ ] Voice agent (German first; Italian greeting line optional in Meran) —
      fastest stack for v1: a voice-agent platform (e.g. Retell/Vapi/ElevenLabs
      Agents) + a +39 number
- [ ] Calendar: Cal.com (free tier) as v1 — gives a booking API for the agent,
      a booking page for the website later, and built-in reminders
- [ ] Conditional forwarding on 0473 445377:
      `**61*<AI-Nummer>#` (no answer, ~15 s) and `**67*<AI-Nummer>#` (busy);
      off: `##61#` / `##67#`. Also forward **when closed** (Mo/So, after 19:00)
      — his dead days become booking days.
- [ ] WhatsApp win message to André per booking + summary per unresolved call
- [ ] Test together in the salon: he calls his own number, lets it ring out,
      books a cut, sees the WhatsApp arrive. That moment closes the deal.

### Agent scope for the test (keep it small)

- CAN: book/reschedule/cancel cuts & standard services, answer prices/hours,
  offer same-day slots, take messages
- CANNOT: color consultations (offer callback), complaints (callback),
  anything uncertain → message + WhatsApp summary. Never guess. Never book wrong.
- Every booking ends with: „Soll ich dir gleich deinen nächsten Termin
  in 6 Wochen eintragen?"

### What we measure (= the case study)

| Metric | Target (30 days) |
|---|---|
| Calls handled by AI | count all; expect 20–40 given solo + 2 closed days |
| Appointments booked by AI | **≥ 10** (guarantee threshold) |
| € recovered (bookings × real prices) | 300–600 € expected |
| Bookings on Mo/So/after-hours | highlight separately — pure found money |
| Fallbacks/messages taken | quality check, review worst calls weekly |
| Hang-ups | < 20 % of AI-answered calls |

Weekly: listen to the 3 worst calls, fix prompts/data. Day 30: one-page recap
in euros → convert to paid → ask for the 2 intros.

## Implement-everything phase (Day 30+, after conversion)

1. **Publish the website** (his real prices, verbatim Google review quotes,
   his photos/logo, Impressum + Datenschutz) — booking page goes live on it,
   "Online buchen" teaser becomes real
2. **No-show reminders** via WhatsApp (24 h before, JA/verschieben)
3. **Review requests** day after each appointment → grow the 4.8★ base
4. **Weekly digest** replaces per-booking pings (keep milestones)
5. **Waitlist backfill** on cancellations (later)

## Risks / notes

- His mobile 339 585 8111 may be where calls actually land — confirm which
  number rings in the salon and put forwarding on THAT one.
- Meran is bilingual: agent greets in German, must gracefully handle Italian
  callers (v1: Italian greeting + „posso continuare in italiano" if the
  platform supports it; otherwise polite handoff → message)
- AI discloses itself in the first sentence (EU AI Act)
- Track our per-call cost from day 1 — this pilot sets the margin baseline
