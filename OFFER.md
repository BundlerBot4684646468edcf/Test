# The Offer — AI Receptionist for Hair Salons (DACH)

Working name for the offer: **"Nie wieder ein verpasster Termin"**

This document is the complete offer playbook: the promise, the package, pricing,
the guarantee, every customer-facing script and message template (German), the
voice-agent call flow, and the build + onboarding checklists.

---

## 1. The Core Promise

> **"Du verlierst keinen Anruf mehr — wir buchen die Termine, die dir bisher
> entgangen sind."**

Everything is denominated in **euros of recovered revenue**, never in "calls
answered" or "AI features".

### The ROI math (use in every pitch)

```
verpasste Anrufe/Woche  ×  Buchungsquote  ×  Ø Bon  ×  4,3  =  verlorener Umsatz/Monat
```

Example for the pitch:

| | |
|---|---|
| Verpasste Anrufe pro Woche | 8 |
| davon hätten gebucht | ~60 % → ≈ 5 |
| Durchschnittsbon | 50 € |
| **Verlorener Umsatz pro Monat** | **≈ 1.075 €** |
| Preis des Angebots | 199 €/Monat |
| **Netto zurückgeholt** | **≈ +875 €/Monat** |

Secondary math (the Fresha angle): every new client who books through the
salon's **own** phone/website instead of the Fresha marketplace saves the salon
**20 % of the first ticket** (Fresha marketplace new-client commission, min.
~6 €). Ten new clients a month ≈ 100 €+ saved — the website bonus alone can
cover half the subscription.

---

## 2. The Offer Stack (what the salon gets)

| # | Component | What it is | Role in the offer |
|---|-----------|------------|-------------------|
| 1 | **KI-Telefonassistentin** | Answers on conditional forwarding: only when the salon doesn't pick up within ~15 s, plus after hours. Native-quality German (Italian optional). Books, reschedules, cancels, answers prices/hours. | The core product |
| 2 | **Terminbuchung direkt in den Kalender** | Fresha integration if they have it; **our own booking page** if they have nothing. | Makes the call end in revenue |
| 3 | **Kostenlose Salon-Website** | Auto-generated from their Google Business profile, booking page embedded, their logo + colors. | Lead magnet / bonus |
| 4 | **No-Show-Erinnerungen** | WhatsApp/SMS 24 h before each appointment with one-tap confirm/reschedule. | Rounds out "no more lost time" |
| 5 | **Gewinn-Nachrichten + Monatsreport** | WhatsApp ping at every win ("Termin gebucht ≈ 45 €"), weekly/monthly recap in euros. | Retention + referrals |
| 6 | **Anruf-Protokolle** | Every call transcribed and visible; unanswerable calls → message taken → WhatsApp summary to owner. | Trust |

**Positioning rule:** items 3–6 are presented as *bonuses* around item 1+2.
Never sell the website as the product — it's the gift that opens the door.

---

## 3. The Guarantee

> **"Wenn wir im ersten Monat nicht mindestens 10 Termine für dich buchen,
> zahlst du nichts."**

Why this is safe for us: with conditional forwarding we only ever touch calls
the salon was **already losing**. We cannot make them worse off, so the
guarantee almost never triggers — but it removes 100 % of the buyer's risk and
is the loudest line in the pitch. Calibrate the number (10) after the first
3 pilot salons; it should be a threshold we hit ~95 % of the time.

---

## 4. Pricing

| Item | Price | Notes |
|------|-------|-------|
| **Einrichtung (setup)** | **149 € einmalig** | Covers the onboarding call, forwarding setup, website generation. Creates commitment; waive it in pilot deals as a negotiation chip, never discount the monthly. |
| **Monatlich pro Salon** | **199 €/Monat** | Flat. Fair-use cap (~400 handled calls/month) in the fine print, never per-minute pricing. |
| **Jahreszahlung** | **2 Monate geschenkt** (1.990 €/Jahr) | Push after month 2–3, once trust exists. Annual payers don't churn. |
| Pilot (first 3–5 salons) | 30 Tage kostenlos, dann 149 €/Monat lifetime | In exchange for: a testimonial, a case-study one-pager, and 2 referral intros. Put that in writing. |

**Internal margin watch:** track per-salon telephony + LLM + WhatsApp API cost
monthly. One chatty salon on a flat plan can eat the margin — know it before it
does. WhatsApp Business API is billed per conversation; reminders and win
messages are cheap but not free.

---

## 5. Sales Assets

### 5.1 The Missed-Call Audit (foot in the door)

**Variant A — the live test:** call the salon Tuesday ~11:00 and Saturday
~14:00. If nobody picks up, that recording *is* the pitch.

**Variant B — the 1-week measurement (stronger):** offer to route their
missed calls for one week to a number that only counts and records. Then show
the tally in euros. You sell against their own data — nearly unanswerable.

**Audit pitch script (German, phone or walk-in):**

> „Hallo, hier ist Alex von [Firma]. Ich mach's kurz, ich weiß, bei euch
> klingelt's ständig — genau darum rufe ich an. Ich habe euch letzten Samstag
> um 14 Uhr angerufen und niemand konnte rangehen. Völlig normal, ihr habt
> Kundinnen auf dem Stuhl. Aber jeder verpasste Anruf ist im Schnitt ein
> 50-Euro-Termin, der zum Salon nebenan geht.
>
> Mein Angebot: Wir messen eine Woche lang kostenlos, wie viele Anrufe bei euch
> durchrutschen. Ihr müsst nichts umstellen, es kostet nichts, und danach zeige
> ich euch schwarz auf weiß, wie viel Umsatz da liegt. Wenn's nichts ist —
> auch gut, dann wisst ihr's wenigstens. Wann passt euch ein 10-Minuten-Termin?"

**Objection handling:**

- *„Wir haben schon Fresha."* → „Perfekt — wir buchen direkt in euren
  Fresha-Kalender. Und: Termine über euer eigenes Telefon kosten euch keine
  20 % Marketplace-Provision."
- *„Eine KI am Telefon? Das mögen meine Kundinnen nicht."* → „Ruf sie selbst
  an: [Demo-Nummer]. Und sie geht nur ran, wenn ihr es nicht könnt — die
  Alternative ist nicht ‚Mensch statt KI', sondern ‚KI statt niemand'."
- *„Zu teuer."* → zurück zur Rechnung: „Ihr verliert gerade ~1.000 € im Monat.
  Das hier kostet 199 €. Und der erste Monat ist garantiert: keine 10 Termine,
  keine Rechnung."
- *„Ich muss drüber nachdenken."* → „Klar. Lass uns nur die kostenlose Messung
  starten — dann denkst du mit echten Zahlen nach statt mit meinen."

### 5.2 The Demo Number

A phone number anyone can call and book a fake appointment on the spot, in
German, in under 90 seconds. **The demo is worth ten slide decks.** Print it on
everything. During a pitch: hand the owner *their own phone*.

### 5.3 One-Pager (print/PDF, German) — copy

> **Nie wieder ein verpasster Termin.**
>
> Während du schneidest, klingelt das Telefon. Jeder verpasste Anruf ist im
> Schnitt ein 50-€-Termin — der zum Salon nebenan geht.
>
> **So funktioniert's:** Geht ihr nicht ran, übernimmt nach 15 Sekunden unsere
> KI-Assistentin. Sie klingt natürlich, kennt eure Preise und euer Team, und
> bucht den Termin direkt in euren Kalender. Ihr bekommt sofort eine
> WhatsApp: „Neuer Termin: Balayage, Sa 11:00, ≈ 120 €."
>
> **Im Paket:** KI-Telefonassistentin · Terminbuchung (Fresha oder eigene
> Buchungsseite) · kostenlose Salon-Website · No-Show-Erinnerungen per
> WhatsApp · Monatsreport in Euro.
>
> **Garantie:** Mindestens 10 gebuchte Termine im ersten Monat — sonst
> zahlt ihr nichts.
>
> **199 €/Monat.** Einrichtung 149 € einmalig. Keine Vertragsbindung im
> ersten Jahr. — Demo: einfach anrufen: **[Demo-Nummer]**

### 5.4 Case-study one-pager (after 60 days per salon)

Template to fill: calls answered / appointments booked / € recovered /
new Google reviews / one owner quote / photo. This is the sales machine for
customers 4–20. Instrument everything from day one so this takes 10 minutes
to produce.

---

## 6. Message Templates (WhatsApp, German)

**Win message — per booked appointment (weeks 1–4, then auto-downgrade to weekly):**

> 🎉 Neuer Termin gebucht, während ihr im Salon beschäftigt wart:
> **Damenhaarschnitt, Fr 14:30 bei Sandra** — geschätzter Umsatz **45 €**.
> Das war Anruf Nr. 12 diesen Monat, den wir aufgefangen haben.

**Milestone message:**

> 🏆 Meilenstein: **10 Termine** diesen Monat über eure KI-Assistentin gebucht
> — zusammen ca. **520 €**, die sonst weg gewesen wären.

**Weekly digest (from month 2):**

> 📊 Eure Woche: **23 Anrufe angenommen**, davon **9 Termine gebucht**
> (≈ 430 €), 3 Rückrufwünsche weitergeleitet, **+2 Google-Bewertungen** ⭐️.
> Details: [Link]

**No-show reminder (to the client, 24 h before):**

> Hallo {Vorname}! Erinnerung an deinen Termin morgen um **{Uhrzeit}** bei
> **{Salon}** ({Leistung} bei {Mitarbeiterin}). Passt alles? 👍 Antworte kurz
> mit JA — oder verschiebe hier: {Link}

**Owner summary — call the AI couldn't resolve:**

> 📞 Anruf um 16:42 — Frau Berger möchte eine **Farbkorrektur** besprechen und
> bittet um Rückruf: **0664 123 45 67**. Zusammenfassung: [2 Sätze].
> Aufnahme/Transkript: [Link]

**Review request (to the client, 1 day after appointment):**

> Hallo {Vorname}, danke für deinen Besuch bei {Salon} gestern! 💇‍♀️ Wenn du
> zufrieden warst, würde uns eine Google-Bewertung riesig helfen (dauert
> 30 Sekunden): {Google-Review-Link}

---

## 7. Voice Agent — Call Flow Spec (v1)

```
1. GREETING      „{Salon}, hier ist die digitale Assistentin — grüß Gott!
                  Wie kann ich helfen?"
                  (EU AI Act: AI identity is disclosed in the first sentence.)

2. INTENT        buchen | verschieben | absagen | Preisfrage | Öffnungszeiten |
                 Sonstiges

3a. BOOK         Leistung → Wunschtermin/Zeitraum → (optional) Mitarbeiterin
                 → Kalender-Lookup → max. 2–3 Slot-Vorschläge → Name + Handynummer
                 → bestätigen → Kalendereintrag (Fresha API / eigene Buchungsseite)
                 → SMS/WhatsApp-Bestätigung an Kundin → Win-Message an Inhaberin

3b. RESCHEDULE/  Termin über Name + Handynummer finden → neuen Slot anbieten
    CANCEL       → bei Absage: Warteliste anpingen (Slot-Backfill)

3c. PRICE/HOURS  Antwort aus Onboarding-Daten. Bei Spannen ehrlich sein:
                 „Balayage liegt je nach Haarlänge zwischen 120 und 180 € —
                 soll ich dir einen Beratungstermin eintragen?"

3d. FALLBACK     Alles, was nicht sicher lösbar ist (Reklamation, Farbkorrektur,
                 Stammkundin mit Spezialwunsch):
                 „Das klärt {Inhaberin} am besten persönlich — ich richte es aus,
                  sie ruft dich zurück."
                 → Nachricht aufnehmen → WhatsApp-Summary an Inhaberin.
                 NIE raten. NIE falsch buchen. Ein eleganter Handoff ist ok,
                 eine selbstbewusst falsche Buchung zerstört das Vertrauen.

4. CLOSE         Nach jeder Buchung IMMER: „Soll ich dir gleich deinen nächsten
                 Termin in 6 Wochen eintragen?" (Rebooking = Umsatzhebel Nr. 1)
```

**Quality bar:** native-quality German voice, < 1 s response latency. A robotic
or laggy German voice is the #1 demo killer in DACH and the moat against
US-built competitors.

**KPIs per salon (weekly review of the worst calls = the real roadmap):**
containment rate, booking conversion, hang-up rate, fallback rate, latency.

---

## 8. Onboarding (goal: live in < 1 hour, done FOR them)

**Step 0 — before first contact (lead magnet):** generate the preview website
from the **Google Business URL alone** (name, hours, photos, reviews, services
are public). Outreach says „wir haben sie schon gebaut" — claiming it costs the
owner exactly **one field** (phone or email). The 10-field form below comes
*after* they bite, filled in BY US in a call.

**Step 1 — onboarding call (20–30 min, covered by the 149 € setup fee):**

- [ ] Salonname, Inhaber:in, Telefon, E-Mail
- [ ] Google Business URL (usually already have it)
- [ ] Öffnungszeiten
- [ ] Services + Preise (incl. ranges: „ab", „je nach Haarlänge")
- [ ] Mitarbeiternamen (+ wer was macht, wer buchbar ist)
- [ ] Buchungssystem: Fresha / nichts / anderes
      → **„nichts" = best segment**: they get our booking page and we ARE the
      system. Highest lock-in. Priority, not edge case.
- [ ] WhatsApp-Nummer der Inhaberin (win messages, summaries)
- [ ] Logo + Farben für die Website
- [ ] Impressum-Daten + Datenschutz-Verantwortliche:r (legally required, DE/AT)
- [ ] Einwilligung Anrufaufzeichnung + Kundennachrichten (GDPR)

**Step 2 — technical go-live (15 min):**

- [ ] Assign the salon's AI number
- [ ] Conditional forwarding on the salon phone — laminated card with carrier
      codes (Telekom / Vodafone / A1 / Magenta):
      `**61*<KI-Nummer>#` (bei Nichtannahme), `**67*<KI-Nummer>#` (bei besetzt)
      Deactivate: `##61#` / `##67#` — the owner can switch it off herself
      anytime. Say that out loud; it kills the fear.
- [ ] Test call together, book a test appointment, show the WhatsApp ping
- [ ] Website live on subdomain (own domain = later upsell)

**Rollout arc per salon:** Week 1 overflow-only → win messages firing → Week 3
add no-show reminders + review requests → Month 2 weekly digest + annual-plan
pitch → Day 60 case-study one-pager + ask for 2 referrals.

---

## 9. Build Checklist (tech, in priority order)

**Must exist before selling (v1):**

- [ ] Voice agent DE: greeting/disclosure, intents, booking flow, fallback → message
- [ ] Telephony number provisioning + conditional-forwarding playbook
- [ ] One booking path: Fresha API **or** own booking page (build own page first
      if Fresha API access is slow — it also serves the „nichts" segment)
- [ ] WhatsApp Business API sender: win messages, owner summaries, confirmations
- [ ] Call log with transcripts (a simple list is enough for v1)
- [ ] Demo number with a fictional salon
- [ ] Missed-call measurement mode (count + record only) for the audit

**Fast follow (weeks after first pilots):**

- [ ] No-show reminder automation
- [ ] Review-request automation
- [ ] Website auto-generation from Google Business profile
- [ ] Weekly digest + monthly euro report
- [ ] Waitlist backfill on cancellations

**Later (don't build yet):**

- [ ] Dashboard beyond the call list · own-domain websites · more booking-system
      integrations (add per lost deal, not speculatively) · Italian voice ·
      adjacent verticals (barbershops, nails — only once onboarding < 1 h,
      churn survives month 6, and referrals come unasked)

---

## 10. Compliance (DACH — non-negotiable, and a selling point)

- **EU AI Act:** the agent discloses it's an AI in the first sentence. Done in
  the greeting above.
- **GDPR:** consent for call recording/transcripts; processing agreement (AVV)
  with each salon; data stored in the EU; deletion on request.
- **Impressum + Datenschutzerklärung** on every hosted salon website; collected
  during onboarding.
- **WhatsApp:** customer messages only with consent (collected at booking).
- Put „100 % DSGVO-konform, KI transparent gekennzeichnet" on the one-pager —
  it reassures exactly the cautious owner who is otherwise the hardest sale.

---

## 11. The Funnel (end to end)

```
Google-Business-Scrape → auto-generierte Preview-Website
    → Outreach / Missed-Call-Audit („wir haben schon eine Website für euch")
    → Owner claimt die Seite (1 Feld)
    → Onboarding-Call (Formular wird FÜR sie ausgefüllt, 149 € Setup)
    → Rufumleitung aktiv, erste Win-Messages innerhalb von Tagen
    → No-Show-Reminder + Review-Automation dazu
    → Monat 2: Digest + Jahresplan-Pitch
    → Tag 60: Case Study + 2 Referrals
    → Nicht-Konvertierer: Drip-Liste (Win-Stories anderer Salons,
      „deine Seite hatte 40 Besucher diesen Monat")
```
