# Virtual Test Setup — AI Receptionist for hair.style.by andré

Goal: Andre calls a **virtual number** (or taps a web link) and experiences his
own AI receptionist booking a real appointment into a test calendar — before
anything touches his real phone, his Google profile, or a customer.

---

## 1. The two ways to run the virtual test

**Option A — web voice link (recommended for day 1, zero telephony):**
Voice-agent platforms (Vapi, Retell, ElevenLabs Agents) give every agent a
shareable **browser call link/widget**. Andre taps the link on his phone and
talks to the agent immediately. No number, no KYC, no cost per minute, live in
an afternoon. Perfect for the "virtual salon" feel: send him the link together
with the preview website.

**Option B — real virtual number (the phone feel):**
- Platform-provided numbers are instant but **US (+1)** — an Italian mobile
  calling +1 is expensive and feels wrong. Avoid.
- Fast EU alternative: provision a **+43 Austrian or +49 German number** via
  Twilio (light KYC, often same-day) and attach it to the agent. Intra-EU
  calls are capped-cheap from his plan. Fine for a private test.
- The proper **+39 number** (the production front door) needs Italian
  regulatory KYC (days–2 weeks): **start that application in parallel now**,
  it becomes the real install later.

Recommended sequence: Option A this week → +43/+49 number as soon as Twilio
clears it → +39 production number when KYC completes.

## 2. Stack for the test

| Piece | Choice | Why |
|---|---|---|
| Voice agent | Vapi or Retell | Built-in **Cal.com integration** (check availability + book as native tools), web call link, phone attach later |
| Voice | Best available native German voice (e.g. ElevenLabs voice inside the platform) | The demo lives or dies on the German sounding human |
| Calendar | **Cal.com free tier**, event types per service (Herrenschnitt 30 min, Farbe 120 min…) with Andre's opening hours as availability | Real slot logic in the test; later this same calendar powers the website's booking page |
| Win message | For the test: email or a manual WhatsApp is fine; wire the real WhatsApp ping before go-live | The "phone buzzes" moment can be simulated day 1 |

## 3. System prompt (paste into the platform, German)

```
# ROLLE
Du bist die digitale Telefonassistentin von „hair.style.by andré", einem
Friseursalon mit Barbershop-Flair in Meran (Südtirol). Du klingst natürlich,
freundlich und unkompliziert — wie eine gute Rezeptionistin, nicht wie ein
Computer. Du sprichst Deutsch (Du-Form, herzlich, südtirolerisch unaufgeregt).
Wenn jemand Italienisch spricht, wechsle vollständig ins Italienische.

# OFFENLEGUNG (Pflicht, erster Satz)
Beginne jedes Gespräch mit:
„Hair Style by André, hier ist die digitale Assistentin — grüß Gott!
André ist gerade beschäftigt, aber ich kann dir gern einen Termin eintragen.
Wie kann ich helfen?"

# SALON-DATEN
- Adresse: Via Monte Tessa 16/B, 39012 Meran
- Öffnungszeiten: Di–Fr 8:00–19:00, Sa 8:00–16:00, Mo + So geschlossen
- Team: André (Inhaber, schneidet alles selbst)
- Leistungen & Testpreise (für den Test — echte Preisliste folgt):
  Herrenschnitt 28 € (30 Min) · Bart & Kontur 18 € (20 Min) ·
  Damenschnitt ab 42 € (45 Min) · Farbe/Balayage ab 75 € (120 Min) ·
  Styling ab 25 € (30 Min) · Rockabilly-Cuts ab 32 € (45 Min)

# TERMIN BUCHEN (Kernaufgabe)
1. Leistung erfragen (falls unklar: kurz nachfragen, nicht raten)
2. Wunschtag/-zeit erfragen; Kalender prüfen (Tool: Verfügbarkeit)
3. Maximal 2–3 Slot-Vorschläge nennen, nie mehr
4. Name + Handynummer aufnehmen, Nummer zur Sicherheit wiederholen
5. Buchen (Tool: Termin anlegen), dann bestätigen:
   „Passt — {Leistung} am {Tag} um {Uhrzeit} bei André. Du bekommst
   gleich eine Bestätigung per SMS/WhatsApp."
6. IMMER danach fragen: „Soll ich dir gleich deinen nächsten Termin
   in etwa sechs Wochen eintragen?"

# WEITERE FÄLLE
- Verschieben/Absagen: Termin über Name + Nummer finden, neuen Slot anbieten
- Preisfragen: aus der Liste antworten. Bei „ab"-Preisen ehrlich:
  „Das hängt von der Haarlänge ab — {Preis} aufwärts. Soll ich dir einen
  Termin eintragen, dann schaut sich André das an?"
- Öffnungszeiten/Adresse: direkt beantworten
- „Spontan heute noch?": aktiv im Kalender nach heutigen Lücken suchen —
  spontane Termine sind Andrés Markenzeichen

# NIEMALS
- NIEMALS Farbkorrekturen oder Reklamationen selbst klären → „Das bespricht
  André am besten persönlich — ich richte es ihm aus, er ruft dich zurück."
  Name + Nummer + Anliegen aufnehmen.
- NIEMALS einen Termin raten oder doppelt buchen — wenn das Kalender-Tool
  nicht antwortet, Nachricht aufnehmen statt buchen
- NIEMALS medizinische/chemische Beratung (Allergien etc.) → Rückruf
- NIEMALS lange Monologe: Antworten kurz halten, 1–2 Sätze, dann Frage

# GESPRÄCHSENDE
Zusammenfassen, bedanken, freundlich verabschieden:
„Danke dir — bis {Tag} dann, ciao!"
Nach jedem Anruf: Zusammenfassung erzeugen (wer, was, wann, offene Punkte)
für die Nachricht an André.
```

## 4. Cal.com test calendar

- Event types: one per service with the durations above
- Availability = his hours: Di–Fr 8–19, Sa 8–16 (Mo/So none)
- Buffer 5–10 min between bookings
- Connect it in the platform's Cal.com integration → the agent's
  „Verfügbarkeit prüfen"/„Termin anlegen" tools point at it
- Seed a few fake bookings so the calendar looks realistically busy and the
  agent has to negotiate slots — a too-empty calendar makes a boring demo

## 5. Andre's test script (send him this with the link/number)

Ruf an und probier bewusst diese 10 Dinge:

1. Herrenschnitt für nächste Woche buchen — normal durchlaufen lassen
2. „Habt ihr heute noch was frei?" (Spontan-Test — sein Markenzeichen)
3. Einen Termin auf **Montag** verlangen (muss ablehnen + Alternative bieten)
4. Nach dem Preis für Balayage fragen (muss „ab, je nach Haarlänge" sagen)
5. Den gebuchten Termin **verschieben**
6. Auf Italienisch anrufen („Buongiorno, vorrei un appuntamento…")
7. Eine **Farbkorrektur** ansprechen (muss Rückruf anbieten, NICHT buchen)
8. Undeutlich/genuschelt sprechen, Hintergrundlärm (Robustheit)
9. Versuchen, sie aus dem Konzept zu bringen (Smalltalk, Blödsinn)
10. Auflegen mitten im Satz, nochmal anrufen (fängt sie sich sauber?)

Danach zeigst du ihm: die Termine stehen im Kalender, und für jeden Anruf
gibt es Transkript + Zusammenfassung. DAS ist der Moment, in dem aus dem
Test die Rufumleitung wird.

## 6. Pass/fail bar for the virtual test

- Booking flow end-to-end without a single wrong booking (10/10 — one wrong
  booking in the demo kills more trust than ten clumsy sentences)
- German sounds natural, latency under ~1.5 s
- Monday correctly refused, color correction correctly deflected
- Italian at least gracefully handled (full Italian flow can come later)
- Every call produces a usable summary

If any of these fail: fix, re-run yourself, THEN re-demo to Andre. Never let
him hit a known bug.

## 7. After he's convinced (bridge to the real pilot)

1. Real prices + never-book list from the onboarding call → update prompt
2. +39 number live → attach agent → sequential ring (his phone first, 15 s)
3. Google Business number swap → the test becomes PILOT-ANDRE.md, day 1
