# Mundpost — Setup von A bis Z

## Schritt 1️⃣: Google Cloud Console — Google Places API

### 1. Google Cloud Projekt erstellen
- Gehe zu https://console.cloud.google.com/
- Melde dich mit deinem Google-Konto an
- Klick auf **"Create Project"** oben
- Name: `mundpost` (oder beliebig)
- Click **Create**

### 2. Places API aktivieren
- Im Menü links: **APIs & Services** → **Library**
- Suche nach **"Places API"**
- Klick auf das Ergebnis
- Click **ENABLE**

### 3. API-Key erstellen
- Im Menü: **APIs & Services** → **Credentials**
- Click **+ Create Credentials** → **API Key**
- Der Key wird angezeigt (z. B. `AIzaSyD...`)
- **Kopiere diesen Key!**

### 4. In `.env` einfügen
```
GOOGLE_PLACES_API_KEY=AIzaSyD...
```

---

## Schritt 2️⃣: Twilio — SMS Versand

### 1. Twilio Konto erstellen
- Gehe zu https://www.twilio.com/en-us
- Click **Sign Up** (oben rechts)
- Fülle das Formular aus:
  - Name, Email, Passwort
  - Verifiziere deine Email
  - Verifiziere deine Telefonnummer (SMS)

### 2. Telefonnummer kaufen
- In der Twilio Console (https://console.twilio.com/)
- Menü links: **Phone Numbers** → **Manage** → **Buy a Number**
- Wähle dein Land (z. B. `US`, `DE`, etc.)
- Features: SMS ✅
- Klick **Search** und wähle eine Nummer
- Click **Buy** → Zahle (kostet ~$1/Monat)

### 3. Credentials kopieren
- Gehe zu **Account info** (oben im Dashboard)
- Du siehst:
  - **Account SID** (z. B. `AC123...`)
  - **Auth Token** (z. B. `a1b2c3...`)
  - **Phone Number** (deine neue Nummer, z. B. `+12125551234`)

### 4. In `.env` einfügen
```
TWILIO_ACCOUNT_SID=AC123...
TWILIO_AUTH_TOKEN=a1b2c3...
TWILIO_PHONE_NUMBER=+12125551234
```

---

## Schritt 3️⃣: Resend — Email Versand

### 1. Resend Konto erstellen
- Gehe zu https://resend.com/
- Click **Get Started** (oben rechts)
- Melde dich an (mit Email oder GitHub)
- Verifiziere deine Email

### 2. API-Key erstellen
- Im Dashboard: **API Keys** (im Menü links)
- Click **Create API Key**
- Gib einen Namen ein: `mundpost`
- Copy den Key (z. B. `re_123...`)

### 3. In `.env` einfügen
```
RESEND_API_KEY=re_123...
```

---

## Schritt 4️⃣: Lokale Entwicklung — Alles zusammen

### 1. `.env` Datei komplettieren
Öffne `/home/user/Test/.env` und stelle sicher, dass alle 4 Keys eingetragen sind:

```bash
# Database — SQLite, wird als Datei automatisch angelegt. Leer lassen genügt.
DB_FILE=

# Google Places
GOOGLE_PLACES_API_KEY=AIzaSyD...

# Twilio SMS
TWILIO_ACCOUNT_SID=AC123...
TWILIO_AUTH_TOKEN=a1b2c3...
TWILIO_PHONE_NUMBER=+12125551234

# Resend Email
RESEND_API_KEY=re_123...

# Server
NODE_ENV=development
PORT=3000
FRONTEND_URL=http://localhost:3001
```

### 2. Backend starten (Node.js 22+ nötig)
```bash
npm run dev
```
Server läuft auf `http://localhost:3000`

### 3. Frontend starten (neues Terminal)
```bash
cd frontend
npm run dev
```
Frontend läuft auf `http://localhost:3001`

### 4. Setup validieren
```bash
curl http://localhost:3000/api/setup/status
```

Ergebnis sollte sein:
```json
{
  "services": {
    "googlePlaces": { "configured": true, "status": "✅ Ready" },
    "sms": { "configured": true, "status": "✅ Ready" },
    "email": { "configured": true, "status": "✅ Ready" }
  },
  "allConfigured": true
}
```

---

## Schritt 5️⃣: Erstes Business erstellen & testen

### 1. Business via API erstellen
```bash
curl -X POST http://localhost:3000/api/businesses \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Pizza Pizzeria",
    "ownerName": "Marco",
    "timezone": "Europe/Rome"
  }'
```

Response:
```json
{
  "id": "clv7x...",
  "name": "My Pizza Pizzeria",
  ...
}
```
**Speichere die `id`!**

### 2. Google Place suchen
```bash
curl -X POST http://localhost:3000/api/businesses/clv7x.../find-place \
  -H "Content-Type: application/json" \
  -d '{
    "businessName": "My Pizza Pizzeria",
    "address": "Via Roma 1, Bolzano, Italy"
  }'
```

Response:
```json
{
  "success": true,
  "place": {
    "placeId": "ChIJN1blFljjokMR5syQyewjZvQ",
    "name": "My Pizza Pizzeria",
    "rating": 4.7,
    "userRatingsTotal": 42
  },
  "reviewLink": "https://search.google.com/local/writereview?placeid=ChIJN1blFljjokMR5syQyewjZvQ"
}
```

### 3. Foto hochladen
```bash
curl -X POST http://localhost:3000/api/businesses/clv7x.../photo \
  -F "photo=@/path/to/photo.jpg"
```

### 4. Kunden importieren

Erstelle `customers.csv`:
```csv
firstName,phone,email,servedAt
Marco,+393891234567,marco@example.com,2024-01-15
Anna,,anna@example.com,2024-01-14
Paolo,+393891234568,,2024-01-13
```

Upload:
```bash
curl -X POST http://localhost:3000/api/businesses/clv7x.../customers/import \
  -F "file=@customers.csv" \
  -F "source=past"
```

### 5. Review Request erstellen & versenden
```bash
# Kunden-ID abrufen
curl http://localhost:3000/api/businesses/clv7x.../customers

# Review Request für SMS erstellen
curl -X POST http://localhost:3000/api/businesses/clv7x.../review-requests \
  -H "Content-Type: application/json" \
  -d '{
    "customerId": "customer_id_hier",
    "channel": "sms"
  }'

# oder für Email
curl -X POST http://localhost:3000/api/businesses/clv7x.../review-requests \
  -H "Content-Type: application/json" \
  -d '{
    "customerId": "customer_id_hier",
    "channel": "email"
  }'
```

**Nach 1 Stunde:** Cron-Job sendet SMS/Email automatisch!

### 6. Dashboard testen
- Öffne http://localhost:3001
- Login mit deiner Business ID
- Siehe Kunden, Review Requests, Stats

---

## ⚡ Troubleshooting

### "Twilio not configured"
- Check: `TWILIO_ACCOUNT_SID` und `TWILIO_AUTH_TOKEN` in `.env` vorhanden?
- Hast du `/Credentials` richtig kopiert?

### "Google Places API not configured"
- Check: `GOOGLE_PLACES_API_KEY` in `.env`?
- Hast du Places API in Google Cloud aktiviert?

### "Resend not configured"
- Check: `RESEND_API_KEY` in `.env`?
- Ist der Key von https://resend.com/api-keys?

### SMS wird nicht gesendet
- Logs prüfen: `npm run dev` zeigt `✅ SMS sent` oder `❌ SMS error`
- Ist die Telefonnummer gültig? Format: `+1234567890`

### Email wird nicht gesendet
- Logs prüfen: `npm run dev` zeigt `✅ Email sent` oder `❌ Email error`
- Ist die Email-Adresse gültig?

---

## 🎉 Fertig!

Du hast jetzt ein vollständig funktionierendes Google-Review-System mit:
- ✅ Automatischer Kundenverwaltung
- ✅ SMS & Email Versand (mit Foto/Text)
- ✅ Automatische Cron-Jobs (Versand, Erinnerungen, Metrics)
- ✅ Dashboard für Stats & Wochenbericht
- ✅ GDPR Opt-out Handling

**Nächste Schritte:**
- Personalisiere Email/SMS Templates in `src/services/messaging.ts`
- Setze `dailyBatchLimit` pro Business (z. B. 20/Tag)
- Deploye auf Production (Docker, Heroku, Railway, etc.)

