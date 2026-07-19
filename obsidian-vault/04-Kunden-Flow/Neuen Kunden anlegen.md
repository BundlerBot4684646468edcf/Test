---
tags: [mundpost, onboarding, kunde]
---

# Neuen Kunden (Betrieb) anlegen

Gehört zu: [[Mundpost]] · [[Der komplette Ablauf]]

Gemeint ist hier: ein neuer **Betrieb** (z.B. ein Friseursalon), der Mundpost nutzen will — nicht zu verwechseln mit "Customer" im Datenmodell (das sind die *Endkunden des Betriebs*, siehe [[Datenmodell]]).

## 1. Betrieb (Business) anlegen

```
POST http://localhost:3000/api/businesses
Content-Type: application/json

{
  "name": "Salon Anna",
  "ownerName": "Anna Mair",
  "timezone": "Europe/Rome"
}
```
→ Antwort enthält die `id` des Betriebs. Diese `businessId` wird für alle folgenden Schritte gebraucht.

## 2. Google-Bewertungslink finden

```
POST http://localhost:3000/api/businesses/:businessId/find-place
Content-Type: application/json

{ "businessName": "Salon Anna", "address": "Bozen, Südtirol" }
```
→ sucht den Betrieb über Google Places, speichert `googlePlaceId` + `googleReviewLink`.

**Voraussetzung:** `GOOGLE_PLACES_API_KEY` in `.env` gesetzt, siehe [[API Keys besorgen]].

## 3. Inhaberfoto hochladen + Schild-Position

→ siehe [[Foto-Personalisierung]] für die genauen Schritte (Upload, Sign-Box einstellen, Vorschau).

Optional — ohne Foto funktioniert alles trotzdem, nur ohne das personalisierte Bild (reiner Text-Versand).

## 4. Kunden importieren (CSV)

```
POST http://localhost:3000/api/businesses/:businessId/customers/import
Content-Type: multipart/form-data
Feld: file = <CSV-Datei>
Feld: source = "past" | "new"
```

**CSV-Format** (Spaltenüberschriften genau so):
```csv
firstName,phone,email,servedAt
Anna,+393331234567,anna@example.com,2026-07-15
Marco,+393337654321,,2026-07-16
```
- `firstName` und `servedAt` sind Pflicht
- mindestens `phone` **oder** `email` muss gesetzt sein
- `source: "past"` = Altkunden (für einmaligen Nachfass-Import), `"new"` = laufender Betrieb (für den täglichen Zulauf)

Antwort zeigt `imported`-Anzahl und `errors` (Zeile + Grund) für fehlerhafte Zeilen.

## 5. Bewertungsanfrage für einen Kunden auslösen

```
POST http://localhost:3000/api/businesses/:businessId/review-requests
Content-Type: application/json

{ "customerId": "...", "channel": "sms" }
```
(`channel`: `"sms"` oder `"email"`)

Das legt die Anfrage nur mit Status `queued` an — der eigentliche Versand passiert automatisch durch den [[Cron-Jobs|stündlichen Cron-Job]]. Für einen sofortigen Test siehe [[Test-Checkliste]] (die `/api/setup/test-*`-Endpunkte umgehen die Warteschlange).

## 6. Tagesbudget einstellen

`dailyBatchLimit` am Betrieb (Standard: 20/Tag) begrenzt, wie viele Anfragen der Cron-Job pro Tag pro Betrieb rausschickt — schützt vor Spam-Wirkung bei großen Kundenlisten.

## Danach

→ [[Der komplette Ablauf]] beschreibt, was ab hier automatisch passiert (Versand, Erinnerung, Opt-out, Metrik-Tracking).
