# 🚀 Mundpost — In 3 Schritten starten

Kein Docker, keine Datenbank-Installation, keine Prisma-Downloads.
Die Datenbank ist eine einzige Datei (`mundpost.db`), die automatisch entsteht.

## Voraussetzung
- **Node.js 22 oder neuer** (wichtig — die Datenbank braucht das eingebaute SQLite von Node 22).
  Prüfen: `node -v` → muss `v22...` oder höher zeigen.
  Falls nicht: https://nodejs.org/ → LTS installieren.

---

## Schritt 1 — Backend starten

```bash
cd Test          # in den Projektordner
npm install
npm run dev
```

Du siehst:
```
🚀 Mundpost server running on http://localhost:3000
✅ Cron jobs started
```

Das war's — das Backend läuft. Die Datei `mundpost.db` wird automatisch angelegt.

## Schritt 2 — Frontend starten (zweites Terminal)

```bash
cd Test/frontend
npm install
npm run dev
```

Dashboard: http://localhost:3001

## Schritt 3 — Prüfen ob deine API-Keys erkannt werden

```bash
curl http://localhost:3000/api/setup/status
```

Zeigt für jeden Dienst ✅ oder ❌. Keys trägst du in `.env` ein (siehe SETUP.md).
**Wichtig:** Ohne Keys läuft alles trotzdem — nur echtes SMS/E-Mail-Versenden
und die Google-Suche sind dann deaktiviert.

---

## Schnelltest ohne Keys (alles außer echtem Versand)

```bash
# Business anlegen
curl -X POST http://localhost:3000/api/businesses \
  -H "Content-Type: application/json" \
  -d '{"name":"Meine Pizzeria","ownerName":"Marco"}'
# -> gibt eine "id" zurück, die merkst du dir

# Kunden aus CSV importieren (Datei customers.csv mit Kopfzeile:
# firstName,phone,email,servedAt)
curl -X POST http://localhost:3000/api/businesses/DEINE_ID/customers/import \
  -F "file=@customers.csv" -F "source=past"

# Kundenliste ansehen
curl http://localhost:3000/api/businesses/DEINE_ID/customers
```

Im Dashboard (http://localhost:3001) meldest du dich mit der Business-`id` an.

---

## Was jeder Teil macht

| Endpoint | Zweck |
|---|---|
| `POST /api/businesses` | Betrieb anlegen |
| `POST /api/businesses/:id/find-place` | Google-Place + Review-Link (braucht Google-Key) |
| `POST /api/businesses/:id/customers/import` | CSV-Kunden importieren |
| `POST /api/businesses/:id/photo` | Inhaber-Foto hochladen |
| `POST /api/businesses/:id/review-requests` | Bewertungsanfrage einreihen |
| `PATCH .../review-requests/:id/opt-out` | Abmeldung |
| `GET /api/setup/status` | Welche Keys sind aktiv |

Der stündliche Cron-Job verschickt eingereihte Anfragen (wenn Twilio/Resend-Keys
gesetzt sind) und respektiert das tägliche Limit pro Betrieb.
