---
tags: [mundpost, deployment, hosting]
---

# Hosting auf Railway.app

Gehört zu: [[Mundpost]] · [[Lokales Setup]] · [[API Keys besorgen]]

Ziel: Backend + Frontend laufen dauerhaft online (nicht nur auf deinem PC), damit echte Kunden die SMS/E-Mails bekommen und du/der Betrieb das Dashboard von überall öffnen kannst.

**Backend** → Railway (braucht persistenten Speicher für `mundpost.db` + `uploads/`)
**Frontend** → Vercel (für Next.js gebaut, einfachster Weg, kostenlos)

## Voraussetzung

Code muss lokal in `mundpostworking` auf dem neuesten Stand sein (`engines`-Feld für Node 22 und `UPLOADS_DIR`-Unterstützung wurden am 19.07.2026 ergänzt — neuestes Backend-ZIP verwenden).

## Teil 1: Backend auf Railway

### 1. Account + CLI

1. https://railway.app → Account erstellen (Login mit GitHub geht am schnellsten)
2. Railway-CLI installieren:
   ```
   npm install -g @railway/cli
   ```
3. Einloggen (öffnet Browser):
   ```
   railway login
   ```

### 2. Projekt erstellen

Im Terminal, im Ordner `mundpostworking`:
```
railway init
```
→ Namen vergeben, z.B. "mundpost-backend".

### 3. Persistenten Speicher (Volume) einrichten

Im Railway-Dashboard (railway.app → dein Projekt → dein Service):
1. Tab **"Volumes"** → "New Volume"
2. Mount-Pfad: `/data`

Das ist wichtig — ohne Volume wird bei jedem Deploy die SQLite-Datenbank und alle hochgeladenen Fotos gelöscht (Container-Dateisystem ist sonst flüchtig).

### 4. Umgebungsvariablen setzen

Im Dashboard → Tab **"Variables"** → alle Werte aus deiner lokalen `.env` eintragen, plus:
```
DB_FILE=/data/mundpost.db
UPLOADS_DIR=/data/uploads
NODE_ENV=production
PORT=3000
```
`FRONTEND_URL` und `PUBLIC_BASE_URL` erstmal leer lassen — die trägst du nach, sobald Backend und Frontend ihre echten URLs haben (Schritt 6).

### 5. Deployen

```
railway up
```

Nach dem Build zeigt Railway eine URL wie `https://mundpost-backend-production.up.railway.app`. Diese URL merken.

**Test:** `https://DEINE-RAILWAY-URL/health` im Browser öffnen — sollte `{"status":"ok",...}` zeigen.

## Teil 2: Frontend auf Vercel

1. https://vercel.com → Account erstellen
2. Am einfachsten: Vercel-CLI
   ```
   npm install -g vercel
   cd mundpostworking/frontend
   vercel
   ```
   Fragen mit Enter/Standardwerten durchklicken (oder Projekt manuell im Vercel-Dashboard aus dem Ordner erstellen).
3. Vor dem finalen Deploy: Umgebungsvariable in Vercel setzen (Dashboard → Project → Settings → Environment Variables):
   ```
   NEXT_PUBLIC_API_URL=https://DEINE-RAILWAY-URL/api
   ```
4. Neu deployen (`vercel --prod` oder über Dashboard "Redeploy"), damit die Variable greift.

Vercel gibt dir eine URL wie `https://mundpost-frontend.vercel.app`.

## Teil 3: Beide URLs verbinden

Zurück im Railway-Dashboard (Backend) → Variables ergänzen:
```
FRONTEND_URL=https://mundpost-frontend.vercel.app
PUBLIC_BASE_URL=https://DEINE-RAILWAY-URL
```
Danach Backend neu deployen, damit CORS die Frontend-URL akzeptiert (`src/index.ts` nutzt `FRONTEND_URL` für die CORS-Freigabe).

## Danach: wie lokal, nur online

- `/api/setup/status`, `/api/setup/test-photo/:name` usw. funktionieren genauso, nur unter der Railway-URL statt `localhost:3000`
- Cron-Jobs (stündlicher Versand, täglicher Bewertungs-Check) laufen automatisch weiter, solange der Railway-Service läuft — kein zusätzliches Setup nötig
- Künftige Backend-Updates: nicht mehr ZIP + manuell kopieren, sondern einfach `railway up` im aktualisierten `mundpostworking`-Ordner erneut ausführen

## Kosten (Stand 2026)

- Railway: nutzungsbasiert, Hobby-Plan ca. 5$/Monat Guthaben inklusive, Volume + Backend-Traffic für ein kleines Projekt bleibt meist darunter oder knapp drüber
- Vercel: kostenlos im Hobby-Tier für dieses Projektvolumen völlig ausreichend

## Wichtig: Zugangsdaten

`TWILIO_*`, `RESEND_API_KEY`, `GOOGLE_PLACES_API_KEY`, `R2_*` — für Produktion **eigene/aufgeladene** Konten verwenden, nicht die Trial-Limits aus der lokalen Testphase (siehe [[API Keys besorgen]], [[SMS Zustellprobleme Italien]]).
