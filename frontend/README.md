# Mundpost Frontend

Dashboard für Mundpost Google Review Service, gebaut mit Next.js + React + Tailwind.

## Setup

```bash
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend läuft auf `http://localhost:3001`.

## Struktur

- `app/page.tsx` — Login-Seite
- `app/dashboard/` — Dashboard mit Stats
- `app/dashboard/customers/` — Kundenliste + CSV-Import
- `app/dashboard/reviews/` — Review-Requests Übersicht
- `lib/api.ts` — API-Client mit Axios

## API Integration

Der API-Client in `lib/api.ts` verbindet sich mit dem Backend unter `NEXT_PUBLIC_API_URL` (default: `http://localhost:3000/api`).

Alle Requests schicken automatisch die `X-Business-ID` aus `localStorage`.
