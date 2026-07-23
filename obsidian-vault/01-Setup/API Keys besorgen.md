---
tags: [mundpost, setup, api-keys]
---

# API Keys besorgen

Gehört zu: [[Mundpost]] · [[Lokales Setup]]

## 1. Google Places API (für Google-Bewertungslink)

1. https://console.cloud.google.com/ → Projekt anlegen
2. "Places API" aktivieren
3. Zugangsdaten → API-Key erstellen
4. In `.env`: `GOOGLE_PLACES_API_KEY=...`

Wird gebraucht, um automatisch den `googleReviewLink` für einen Betrieb zu finden (`POST /api/businesses/:id/find-place`).

## 2. Twilio (SMS)

1. https://console.twilio.com/ → Account erstellen
2. **Trial-Konto**: kostenlos, aber:
   - nur an **verifizierte** Zielnummern (Nummer manuell in Twilio-Konsole verifizieren)
   - **max. 5 SMS/Tag** (Fehlercode `63038` wenn überschritten)
   - Absendernummer ist eine US-Nummer → Zustellung nach Italien teils unzuverlässig
3. Account SID + Auth Token aus dem Dashboard kopieren
4. Twilio-Telefonnummer kaufen/zuweisen
5. In `.env`:
   ```
   TWILIO_ACCOUNT_SID=AC...
   TWILIO_AUTH_TOKEN=...
   TWILIO_PHONE_NUMBER=+1...
   ```

**Für Produktion:** Konto aufladen (kein Trial mehr) + idealerweise eine europäische/italienische Nummer oder Alphanumeric Sender ID, sonst bleibt die Zustellung nach Italien unzuverlässig. Siehe [[SMS Zustellprobleme Italien]].

## 3. Resend (E-Mail)

1. https://resend.com/api-keys → Account, API-Key erzeugen
2. In `.env`: `RESEND_API_KEY=re_...`
3. Kostenloses Kontingent reicht für Tests locker

E-Mail ist aktuell der **zuverlässigere Kanal** (im Vergleich zu MMS), weil das Foto direkt inline im HTML eingebettet wird — keine Abhängigkeit von Twilio-MMS-Limitierungen.

## 4. Cloudflare R2 (optional, nur für Foto-**MMS**)

Nur nötig, wenn Fotos per SMS/MMS an echte Kunden verschickt werden sollen (nicht nur E-Mail). R2 gibt dem Foto eine öffentliche `https://`-URL, die Twilio abrufen kann.

1. https://dash.cloudflare.com/ → R2 aktivieren
2. Bucket erstellen, "Public Access" aktivieren
3. Access Key + Secret erzeugen
4. In `.env`:
   ```
   R2_ACCOUNT_ID=...
   R2_ACCESS_KEY_ID=...
   R2_SECRET_ACCESS_KEY=...
   R2_BUCKET_NAME=...
   R2_PUBLIC_BASE_URL=https://pub-xxxx.r2.dev
   ```

Ohne R2: Fotos werden lokal gespeichert (`/uploads`), funktionieren für E-Mail und Dashboard-Vorschau, aber **nicht** für MMS (Twilio kann `localhost`-URLs nicht erreichen).

## Status-Check im Browser

Nach dem Eintragen: `http://localhost:3000/api/setup/status` zeigt, welche Services erkannt wurden (✅/❌).
