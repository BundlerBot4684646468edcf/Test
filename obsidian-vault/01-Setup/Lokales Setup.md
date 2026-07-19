---
tags: [mundpost, setup]
---

# Lokales Setup

Gehört zu: [[Mundpost]]

## Voraussetzungen

- **Node.js 22+** (für `node:sqlite`) — prüfen mit `node -v`
- Windows/Mac/Linux, kein Docker, keine Datenbank-Installation nötig

## Ordnerstruktur

```
mundpostworking/              ← Backend (Hauptordner)
├── src/                      ← Express-Server-Code
├── package.json
├── package-lock.json
├── uploads/                  ← hochgeladene Fotos (lokal)
├── mundpost.db               ← SQLite-Datenbank (entsteht automatisch)
├── .env                      ← eigene Zugangsdaten (NICHT committen)
└── frontend/                 ← Next.js Dashboard (separates npm-Projekt!)
```

⚠️ **Wichtig:** Backend-Updates (ZIPs) immer in den **Hauptordner** `mundpostworking`, NICHT in `frontend/`.

## Backend starten

```bash
cd mundpostworking
npm install
npm run dev
```

Läuft dann auf `http://localhost:3000`. Health-Check: `http://localhost:3000/health`

## Frontend starten

```bash
cd mundpostworking/frontend
npm install
npm run dev
```

Läuft auf `http://localhost:3001`.

**Reihenfolge beim Neustart:** immer erst Backend (3000), dann Frontend (3001) starten. Falls das Dashboard "alt" aussieht → wahrscheinlich läuft noch ein alter Prozess auf einem der Ports. Siehe [[Troubleshooting]].

## .env Datei anlegen

Kopiere `.env.example` zu `.env` und trage Zugangsdaten ein (siehe [[API Keys besorgen]]):

```
DB_FILE=
GOOGLE_PLACES_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
RESEND_API_KEY=
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_PUBLIC_BASE_URL=
NODE_ENV=development
PORT=3000
FRONTEND_URL=http://localhost:3001
```

Ohne R2-Zugangsdaten läuft alles trotzdem — Fotos werden dann einfach lokal gespeichert und per `/uploads/...`-URL ausgeliefert (reicht für E-Mail-Vorschau und Tests, nicht für MMS an echte Kundenhandys).

## Nach jedem Backend-Update (ZIP)

1. ZIP entpacken (landet meist in eigenem Unterordner)
2. `src`, `package.json`, `package-lock.json` in `mundpostworking` reinkopieren, ersetzen
3. `npm install` (wichtig, falls neue Abhängigkeiten dazugekommen sind!)
4. `npm run dev` neu starten

→ Details: [[Test-Checkliste]]
