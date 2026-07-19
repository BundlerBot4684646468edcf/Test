---
tags: [mundpost, sms, italien]
---

# SMS-Zustellprobleme nach Italien

Gehört zu: [[Mundpost]] · [[API Keys besorgen]] · [[Troubleshooting]]

## Das Problem

Twilio-Nummern aus den USA (Standard beim Anlegen eines Trial-Kontos) werden von italienischen Mobilfunknetzen häufig als Spam gefiltert oder gar nicht zugestellt (`status: undelivered`, Fehlercode `30008`).

## Warum

Italienische Carrier bevorzugen/erfordern lokale oder zumindest europäische Absendernummern für SMS-Zustellung. Eine US-Langnummer als Absender ist für viele Filter ein Spam-Signal.

## Lösungsoptionen für Produktion

1. **Italienische/europäische Twilio-Nummer** kaufen (statt der US-Default-Nummer)
2. **Alphanumeric Sender ID** (z.B. "Mundpost" statt einer Nummer als Absender) — von manchen Ländern/Carriern unterstützt, muss bei Twilio beantragt/freigeschaltet werden
3. Twilio-Konto **aufladen** (kein Trial mehr) — Trial-Konten haben ohnehin harte Limits (5/Tag, nur verifizierte Nummern)

## Bis dahin: E-Mail als Hauptkanal

Die aktuelle Implementierung (`src/services/messaging.ts`, `src/services/reviewQueue.ts`) unterstützt beide Kanäle gleichwertig. Da E-Mail (via Resend) zuverlässig zustellt und das personalisierte Foto einfach inline im HTML einbettet (keine MMS-Limitierungen), ist die pragmatische Empfehlung:

> **Für Kunden mit Telefonnummer UND E-Mail: `channel: "email"` bevorzugen**, bis eine lokale Absendernummer eingerichtet ist.

## Diagnose-Tools

- `GET /api/setup/test-sms-check/:number` — sendet + pollt Status in einem Aufruf
- `GET /api/setup/sms-status/:sid` — Status eines bereits gesendeten SMS nachträglich abfragen
- Fehlercode-Übersicht direkt im Code kommentiert: `src/routes/setup.ts`

| Code | Bedeutung |
|---|---|
| 21608 | Trial: Zielnummer nicht verifiziert |
| 30008 | Vom Netz nicht zugestellt (typisch US→Italien) |
| 30007 | Vom Carrier als Spam gefiltert |
| 63038 | Trial-Tageslimit (5 SMS) erreicht |
