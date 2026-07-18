# Mundpost — Google Review Service

Ein Google-Review-Service für lokale Betriebe in Südtirol. Betriebe laden ihre Kundenliste hoch, das System schickt automatisch persönliche SMS/E-Mail-Anfragen mit direktem Link zur Google-Bewertung.

## Setup

### 1. Installation

```bash
npm install
```

### 2. Datenbank starten

```bash
docker-compose -f docker-compose.dev.yml up -d
```

### 3. Prisma Migrations

```bash
npm run prisma:migrate
```

### 4. Server starten

```bash
npm run dev
```

Der Server läuft auf `http://localhost:3000`.

## Schritte

- [x] Schritt 1: Prisma-Schema + Backend-Grundgerüst ✅
- [ ] Schritt 2: Google Places Integration
- [ ] Schritt 3: CSV-Import
- [ ] Schritt 4: Foto-Upload
- [ ] Schritt 5: Versand-Engine (Twilio/Resend)
- [ ] Schritt 6: Opt-out Handling
- [ ] Schritt 7: Dashboard

## Environment-Variablen

Kopiere `.env.example` zu `.env` und fülle die API-Keys aus:

- `GOOGLE_PLACES_API_KEY` — Google Cloud Console
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` — Twilio
- `RESEND_API_KEY` — Resend
- `R2_*` — Cloudflare R2 (optional für Fotos)
