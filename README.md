# 🚀 Mundpost — Google Review Service

Automatisierter Google-Review-Service für lokale Betriebe. Lädt Kundenlisten hoch, sendet personalisierte SMS/Email-Anfragen mit Google-Review-Link, erinnert höflich nach, zeigt Dashboard mit Stats.

## ✨ Features

- 📱 **SMS mit Foto (MMS)** — Persönalisierte SMS mit Inhaber-Porträt
- 📧 **Email mit HTML** — HTML-Templates mit inline Foto & personalisierter Text
- 🔄 **Automatische Versand-Engine** — Cron-Jobs für Versand & Erinnerungen
- 📊 **Dashboard** — Real-time Stats, Wochenbericht mit Chart
- 📋 **CSV Import** — Schnell Kunden importieren
- 🛡️ **GDPR Opt-out** — Abmeldungen per SMS/Email
- ⏱️ **Rate Limiting** — Tägliche Versand-Limits pro Business
- 📈 **Review Tracking** — Automatische Google Places Metrics

## 🏗️ Tech Stack

| Layer | Tech |
|-------|------|
| **Frontend** | Next.js, React, Tailwind CSS, Recharts |
| **Backend** | Node.js, TypeScript, Express, Prisma |
| **Database** | PostgreSQL, Docker |
| **SMS** | Twilio API |
| **Email** | Resend API |
| **Places** | Google Places API |

## 📖 Quick Start

**Neu hier?** Starten mit 5-Minuten-Anleitung:
```bash
# siehe QUICKSTART.md
```

**Detailliertes Setup** (mit API-Key-Anleitung):
```bash
# siehe SETUP.md
```

## 🚦 Jetzt bereit zum Start

Alles ist vorbereitet. Du brauchst nur noch 3 API-Keys:

### 1️⃣ Google Places API Key
→ https://console.cloud.google.com/

### 2️⃣ Twilio Credentials (SMS)
→ https://console.twilio.com/

### 3️⃣ Resend API Key (Email)
→ https://resend.com/api-keys

**Siehe SETUP.md für Schritt-für-Schritt Anleitung!**

## 📂 Projektstruktur

```
mundpost/
├── src/                          # Backend (Node.js + TypeScript)
│   ├── routes/                   # API Endpoints
│   │   ├── businesses.ts        # Business Management
│   │   ├── customers.ts         # Customer Import & List
│   │   ├── reviewRequests.ts    # Review Requests
│   │   ├── photos.ts            # Photo Upload
│   │   └── setup.ts             # Configuration Status
│   ├── services/
│   │   ├── googlePlaces.ts      # Google Places Integration
│   │   ├── messaging.ts         # Twilio + Resend
│   │   ├── csvImport.ts         # CSV Processing
│   │   ├── fileStorage.ts       # Photo Upload
│   │   ├── reviewQueue.ts       # Send Queue
│   │   ├── reviewMetrics.ts     # Daily Metrics
│   │   └── cronJobs.ts          # Scheduled Tasks
│   └── index.ts                 # Express Server
│
├── frontend/                     # Next.js Dashboard
│   ├── app/
│   │   ├── page.tsx            # Login
│   │   ├── dashboard/          # Stats & Charts
│   │   ├── customers/          # CSV Upload
│   │   └── reviews/            # Request List
│   └── lib/api.ts              # API Client
│
├── prisma/
│   └── schema.prisma           # Database Models
│
├── docker-compose.dev.yml      # PostgreSQL
├── SETUP.md                    # Full Setup Guide (A-Z)
└── QUICKSTART.md               # 5-Minute Setup
```

## ⚡ Development

```bash
# Install
npm install

# Start PostgreSQL
docker-compose -f docker-compose.dev.yml up -d

# Migrate DB
npm run prisma:migrate

# Start Backend (Port 3000)
npm run dev

# Start Frontend (Port 3001, separate terminal)
cd frontend && npm run dev

# Check Setup Status
curl http://localhost:3000/api/setup/status
```

## 🚀 Production Deployment

```bash
# Build
npm run build

# Start
npm start

# oder Docker:
docker build -t mundpost .
docker run -e DATABASE_URL=... mundpost
```

## 📖 Dokumentation

- **SETUP.md** — Vollständige Anleitung (mit Links für Google Cloud, Twilio, Resend)
- **QUICKSTART.md** — 5-Minuten-Anleitung + Curl-Beispiele
- **prisma/schema.prisma** — Datenmodell
- **src/services/** — Service-Dokumentation

## 📞 Troubleshooting

- **SMS/Email nicht gesendet?** → Logs prüfen: `npm run dev`
- **Keys nicht erkannt?** → Check `.env` Datei
- **Place nicht gefunden?** → Google Places API aktiviert?

→ Siehe **SETUP.md** für Full Troubleshooting

## ✅ Status

**Production Ready** — Warte nur noch auf deine API-Keys!

---

**Branch:** `claude/mundpost-review-service-nimakq`
**Commits:** Steps 1-7 ✅ Alle Features implementiert
