# Deployment: Dashboard + API hosten

Das Repo enthält ein fertiges `Dockerfile` — jeder Anbieter, der Docker
kann, funktioniert. Empfehlung: **Railway** (einfachster Weg) oder Render.

## Vorbereitung (gilt überall)

Environment-Variablen setzen:

| Variable | Pflicht | Zweck |
|---|---|---|
| `SESSION_SECRET` | **ja** (Prod) | signiert Login-Cookies; ohne sie fliegen alle Sessions bei jedem Neustart raus. Irgendein langer Zufallsstring. |
| `FAMULOR_API_KEY` | empfohlen | wenn gesetzt, brauchen die `/famulor/tools/*`-Endpoints den Header `X-API-Key` — denselben Wert in Famulor beim Custom-Tool hinterlegen. |
| `PLATFORM_ADMIN_KEY` | empfohlen | wenn gesetzt, braucht `POST /salons` (neue Salons anlegen) den Header `X-Admin-Key`. |
| `SALON_DATABASE_URL` | optional | Standard im Container: `sqlite:////data/salon.db`. Für Postgres: `postgresql://...` (dann `psycopg2-binary` zu requirements-backend.txt hinzufügen). |

**Wichtig bei SQLite:** ein persistentes Volume auf `/data` mounten,
sonst sind alle Buchungen nach jedem Deploy weg.

## Railway (empfohlen, ~5 Minuten)

1. https://railway.app → "New Project" → "Deploy from GitHub repo" →
   dieses Repo + diesen Branch wählen. Railway erkennt das Dockerfile.
2. Service → **Variables**: `SESSION_SECRET`, `FAMULOR_API_KEY`,
   `PLATFORM_ADMIN_KEY` setzen.
3. Service → **Volumes**: Volume anlegen, Mount Path `/data`.
4. Service → **Settings → Networking**: "Generate Domain" →
   öffentliche URL, z. B. `https://salon-booking.up.railway.app`.

## Render (Alternative)

1. https://render.com → "New +" → "Web Service" → Repo verbinden,
   Runtime "Docker".
2. Env-Vars wie oben; unter "Disks" eine Disk mit Mount Path `/data`
   anlegen (Disks gibt es nicht im Free-Plan — dort stattdessen die
   Postgres-Variante nutzen).

## Nach dem Deploy

1. Ersten Salon anlegen (einmalig, per API):

```bash
curl -X POST https://DEINE-URL/salons \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: DEIN_PLATFORM_ADMIN_KEY" \
  -d @salon.json
```

`salon.json` = Payload wie in `tests/test_api_e2e.py`
(`TEST_SALON_PAYLOAD`), inkl. `"admin_password": "..."`. Lässt man das
Passwort weg, wird eins generiert und **einmalig** in der Antwort
zurückgegeben.

2. Dashboard: `https://DEINE-URL/salons/<slug>/admin` → Login-Seite →
   Passwort → Verwaltung + Kalender.

3. Famulor: Custom-Tools auf `https://DEINE-URL/famulor/tools/...`
   zeigen lassen, Header `X-API-Key` mitgeben. `GET /famulor/tools`
   liefert das Schema.

## Lokal testen wie in Produktion

```bash
docker build -t salon-booking .
docker run -p 8000:8000 \
  -e SESSION_SECRET=dev-secret \
  -v salon-data:/data \
  salon-booking
# http://127.0.0.1:8000/salons/<slug>/admin
```
