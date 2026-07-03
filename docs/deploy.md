# Deployment: Dashboard + API hosten

Zwei fertige Wege liegen im Repo: **Vercel** (Serverless, `vercel.json` +
`api/index.py`) und **Docker** (Railway/Render/eigener Server,
`Dockerfile`).

## Environment-Variablen (gelten überall)

| Variable | Pflicht | Zweck |
|---|---|---|
| `SESSION_SECRET` | **ja** | signiert Login-Cookies. Auf Vercel doppelt wichtig: ohne festes Secret hat jede Serverless-Instanz ein eigenes → Logins funktionieren nicht zuverlässig. Irgendein langer Zufallsstring. |
| `SALON_DATABASE_URL` | **ja auf Vercel** | Postgres-URL (Neon/Vercel Postgres). `postgres://…` wird automatisch akzeptiert. Ohne Angabe: lokales SQLite — auf Vercel **unbrauchbar**, weil das Dateisystem ephemer ist (alle Buchungen weg). |
| `FAMULOR_API_KEY` | empfohlen | wenn gesetzt, brauchen die `/famulor/tools/*`-Endpoints den Header `X-API-Key` — denselben Wert in Famulor beim Custom-Tool hinterlegen. |
| `PLATFORM_ADMIN_KEY` | empfohlen | wenn gesetzt, braucht `POST /salons` (neue Salons anlegen) den Header `X-Admin-Key`. |

## Vercel (~5 Minuten)

**Vorher: Datenbank anlegen** (SQLite geht auf Vercel nicht):
Im Vercel-Dashboard → Storage → "Create Database" → **Neon Postgres**
(Free-Tier reicht) → die `DATABASE_URL` kopieren.

1. https://vercel.com → "Add New… → Project" → dieses GitHub-Repo
   importieren. Als Production-Branch den Branch
   `claude/bot-backend-sms-fixes-y6crmi` wählen (Settings → Git), solange
   nicht nach `main` gemerged ist.
2. Framework Preset: "Other" — Vercel erkennt `api/index.py` +
   `vercel.json` automatisch, kein Build-Command nötig.
3. Environment Variables setzen: `SALON_DATABASE_URL` (die Neon-URL),
   `SESSION_SECRET`, `FAMULOR_API_KEY`, `PLATFORM_ADMIN_KEY`.
4. Deploy → URL z. B. `https://salon-booking.vercel.app`.

Alternativ per CLI (mit `VERCEL_TOKEN`):

```bash
npx vercel --prod --token $VERCEL_TOKEN
```

**Serverless-Hinweise:**
- Die Tabellen werden beim ersten Request automatisch angelegt.
- Mehrere Instanzen können parallel laufen → der Doppelbuchungs-Re-Check
  ist gegen Postgres sehr gut, aber theoretisch nicht 100 % wasserdicht
  (siehe CLAUDE.md, offener Punkt DB-Constraint). Für Pilot-Traffic
  irrelevant.
- Erster Request nach Pause kann durch Cold-Start ~1–2 s brauchen —
  für Famulor-Tool-Calls okay, aber einkalkulieren.

## Railway (Docker-Weg, Alternative)

1. https://railway.app → "New Project" → "Deploy from GitHub repo" →
   Repo + Branch wählen. Railway erkennt das Dockerfile.
2. Variables wie oben setzen (`SALON_DATABASE_URL` weglassen = SQLite).
3. Bei SQLite: Volume mit Mount Path `/data` anlegen, sonst sind
   Buchungen nach jedem Deploy weg.
4. Settings → Networking → "Generate Domain".

## Nach dem Deploy

1. Ersten Salon anlegen (einmalig, per API):

```bash
curl -X POST https://DEINE-URL/salons \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: DEIN_PLATFORM_ADMIN_KEY" \
  -d @salon.json
```

`salon.json` = Payload wie `TEST_SALON_PAYLOAD` in
`tests/test_api_e2e.py`, inkl. `"admin_password": "..."`. Lässt man das
Passwort weg, wird eins generiert und **einmalig** in der Antwort
zurückgegeben.

2. Dashboard: `https://DEINE-URL/salons/<slug>/admin` → Login →
   Verwaltung + Kalender.

3. Famulor: Custom-Tools auf `https://DEINE-URL/famulor/tools/...`
   zeigen lassen, Header `X-API-Key` mitgeben. `GET /famulor/tools`
   liefert das Schema.

## Lokal testen wie in Produktion

```bash
docker build -t salon-booking .
docker run -p 8000:8000 -e SESSION_SECRET=dev-secret -v salon-data:/data salon-booking
# http://127.0.0.1:8000/salons/<slug>/admin
```
