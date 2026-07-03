# Projekt: Multi-Tenant Friseur/Salon-Buchungs-SaaS

Dieses Dokument ist der Kontext-Handoff für jede KI-Session (oder jeden
Entwickler), die hier weiterarbeitet. Es fasst zusammen: was gebaut wird,
warum genau so, was schon existiert, und was als Nächstes fehlt.

## Ziel

Ein SaaS, mit dem mehrere Friseursalons (Mandanten) Termine für
verschiedene Mitarbeiter mit unterschiedlichen Service-Dauern anbieten
können. Kunden buchen per **KI-Telefon-/Chat-Rezeptionist (Famulor)**;
das Salon-Personal verwaltet alles über die eigene Admin-Seite und den
eigenen Kalender.

## Architektur: EIGENE Buchungs-Engine (Cal.com wurde entfernt)

**Historie:** Ursprünglich lief die Verfügbarkeits-/Kalenderlogik über
Cal.com (Team pro Salon, Event-Types pro Service/Mitarbeiter). Auf
Wunsch des Nutzers wurde Cal.com komplett entfernt — wir haben ein
eigenes Backend mit Kalender, also passiert jetzt alles hier im Repo:

```
Kunde ──(Anruf/Chat/WhatsApp)──> Famulor (Voice/Chat AI)
                                      │
                                      ▼  Function-Calling:
                                      │  get_current_datetime / get_availability /
                                      │  book_appointment / cancel_appointment
                                  Unser FastAPI-Backend (salon/)
                                      │
                                      ▼
                              Eigene Buchungs-Engine (salon/booking.py)
                              + eigene DB (Bookings, Öffnungszeiten, ...)
                                      │
                                      ▼
                        Kalender-UI + Admin-UI (selbst gehostet, HTML)
```

Konsequenz: **Keine externen Abhängigkeiten** mehr für die Buchung —
kein API-Key, keine Kosten pro Salon-Team. Dafür sind SMS
(Bestätigung/Reminder) jetzt **nicht mehr abgedeckt** (liefen vorher
über Cal.com-Workflows) → braucht einen eigenen Anbieter (z. B.
Twilio/Seven), siehe offene Punkte. Die Kundennummer wird dafür bereits
E.164-normalisiert an jeder Buchung gespeichert.

## Die Buchungs-Engine (salon/booking.py)

- **Slots**: 15-Minuten-Raster innerhalb der Öffnungszeiten des Salons
  (pro Wochentag konfigurierbar, Standard Mo–Fr 09–18, Sa 09–14, So zu).
  Ein Slot wird angeboten, wenn der Termin (mit der Dauer des jeweiligen
  Mitarbeiters!) noch vor Ladenschluss endet.
- **Dauern**: Service hat eine Standard-Dauer; pro (Mitarbeiter, Service)
  kann eine eigene Dauer gesetzt sein (`EmployeeService.duration_min`,
  NULL = erbt den Standard). "Lena färbt in 45, Tom braucht 90" ist damit
  nativ abgebildet — kein Event-Type-Trick mehr nötig.
- **Puffer**: `Service.buffer_min` (Aufräumzeit) blockiert nach jedem
  Termin; wird beim Buchen **snapshotted** (`Booking.buffer_min`), damit
  spätere Service-Änderungen bestehende Termine nicht umdeuten.
- **Doppelbuchungs-Schutz**: Der Konflikt-Check (Überlappung inkl. Puffer
  beider Seiten) läuft beim Buchen **noch einmal in derselben
  Transaktion** — der klassische Race (Slot wird zwischen Abfrage und
  Buchung weggeschnappt) endet als `SlotUnavailableError` → HTTP 409 mit
  vorlesbarer Begründung für den Bot. (Annahme: ein Server-Prozess;
  SQLite serialisiert Writes. Bei Multi-Prozess-Deployment später DB-Lock
  oder Unique-Constraint ergänzen.)
- **Round-Robin ("egal wer")**: Ohne Mitarbeiter-Wunsch ist ein Slot
  buchbar, sobald *irgendein* qualifizierter Mitarbeiter frei ist. Beim
  Buchen bekommt ihn der an dem Tag am wenigsten ausgelastete freie
  Mitarbeiter (Load Balancing), bei Gleichstand der mit der niedrigeren
  ID. Namentliche Buchungen und "egal wer" teilen sich denselben
  Kalender pro Mitarbeiter → blockieren sich gegenseitig korrekt.
- **Zeiten**: Alle Buchungszeiten sind naive datetimes in der
  **Salon-Zeitzone** (`Salon.timezone`). Eingehende ISO-Zeiten mit
  Offset werden konvertiert; die UIs zeigen Salon-Lokalzeit.

## Datums-Sicherheit ("Samstag war gestern"-Bug, gefixt)

Der Voice-Bot kannte das aktuelle Datum nicht und hat Wochentage/relative
Daten ("morgen", "Samstag") falsch aufgelöst. Dagegen gibt es drei
Schichten:

1. **Tool `get_current_datetime`** (laut Tool-Beschreibung vor jeder
   Datumsinterpretation aufzurufen): heutiges Datum, Uhrzeit, Wochentag
   (DE/EN) in der Salon-Zeitzone plus die Wochentage der nächsten 7 Tage.
2. **`get_availability`**: Zeitraum komplett in der Vergangenheit → Fehler
   mit Klartext-Hinweis, welcher Tag heute ist; nur Start in der
   Vergangenheit → wird auf heute geklemmt. Antwort enthält immer
   `today`/`today_weekday_de`.
3. **`book_appointment`**: Termin-Start in der Vergangenheit → Fehler mit
   demselben Hinweis, es wird nichts gebucht.

## Codebase-Struktur

```
salon/
  config.py         Env-Var: SALON_DATABASE_URL
  db.py              SQLAlchemy Engine/Session (SQLite lokal, austauschbar gegen Postgres)
  models.py          Salon, Employee, Service, EmployeeService, OpeningHours, Booking
  schemas.py         Pydantic-Schemas für die API
  booking.py         DIE Buchungs-Engine: Slots, Konfliktprüfung, Round-Robin, Storno
  onboarding.py      onboard_salon(): Salon + Mitarbeiter + Services + Matrix + Standard-Öffnungszeiten
  famulor_tools.py   FAMULOR_TOOLS Schema (Function-Calling) + Handler (inkl. cancel_appointment)
  calendar_ui.py     Wochen-Kalender (HTML) mit Storno-Button, liest aus eigener DB
  admin.py           Pflege-Logik: Services/Mitarbeiter/Zuordnung/Öffnungszeiten + Buchungs-Guards
  admin_ui.py        Admin-Seite (HTML): Öffnungszeiten, Mitarbeiter, Services, Wer-macht-was-Matrix
  auth.py            Login pro Salon: PBKDF2-Passwörter, HMAC-Session-Cookies, Login-Seite
  main.py            FastAPI-App: POST /salons, GET+POST /salons/{slug}/login, GET .../logout,
                      GET /famulor/tools,
                      POST /famulor/tools/{get_current_datetime,get_availability,
                                           book_appointment,cancel_appointment},
                      GET /salons/{slug}/bookings, DELETE /salons/{slug}/bookings/{id},
                      GET /salons/{slug}/calendar, GET /salons/{slug}/admin,
                      GET /salons/{slug}/config,
                      POST/PATCH/DELETE /salons/{slug}/services[/{id}],
                      POST/DELETE /salons/{slug}/employees[/{id}],
                      PUT /salons/{slug}/qualifications,
                      PUT /salons/{slug}/opening-hours

tests/
  test_salon_onboarding.py   Onboarding + Famulor-Tool-Handler (Datum, Telefon, Storno)
  test_booking_engine.py      Engine-Garantien: Doppelbuchung, Puffer, Öffnungszeiten, Round-Robin, Race
  test_admin.py               Admin-Layer inkl. Guards für bevorstehende Termine
  test_auth.py                Login, Session-Isolation pro Salon, API-Keys
  test_api_e2e.py             E2E über die echten HTTP-Endpoints (FastAPI TestClient)

app.py              UNVERWANDTE Alt-App (Hotel-Reputation-MVP) — nicht anfassen, nicht Teil dieses Projekts
```

`app.py` und `requirements.txt`-Einträge für Streamlit/Plotly/etc. gehören
zu einem völlig anderen, älteren Projekt in diesem Repo und haben mit der
Salon-Buchung nichts zu tun.

**Achtung Schema-Änderung:** Die DB hat kein Migrations-Setup; nach dem
Cal.com-Umbau eine evtl. vorhandene lokale `salon.db` löschen (wird beim
Start neu angelegt).

## Admin-UI: Selbstverwaltung (`GET /salons/{slug}/admin`)

- **Öffnungszeiten** pro Wochentag (geöffnet/zu, von, bis) — wirken sofort
  auf die angebotenen Slots, bestehende Termine bleiben unberührt.
- **Mitarbeiter** anlegen/entfernen. Entfernen = Soft-Delete
  (`active=False`), vergangene Termine bleiben zuordenbar.
- **Services**: Name, Standard-Dauer, Puffer, Preis.
- **"Wer macht was"-Matrix** inkl. eigener Dauer pro Mitarbeiter
  (leer = erbt Standard; Standard-Änderung wirkt dann automatisch mit).
- **Guards**: Jede Lösch-/Abwahl-Aktion, an der noch zukünftige Termine
  hängen, liefert HTTP 409 mit Klartext; das UI fragt nach und wiederholt
  mit `force=true` — die betroffenen Termine werden dann **explizit
  storniert** (sichtbar im Kalender), nie stillschweigend verwaist.

## Auth (salon/auth.py)

- **Login pro Salon**: Passwort wird beim Onboarding gesetzt
  (`admin_password` in `POST /salons`); fehlt es, wird eins generiert und
  **einmalig** in der Antwort zurückgegeben (nur der PBKDF2-Hash wird
  gespeichert). Login-Seite: `GET /salons/{slug}/login`.
- **Session** = HMAC-signierter Cookie (12h), gescoped auf
  `/salons/{slug}` — die Session von Salon A öffnet nie Salon B (wird
  server­seitig geprüft, nicht nur über den Cookie-Pfad). Geschützt sind:
  Admin, Kalender, Config, Buchungsliste/-storno und alle
  Pflege-Endpoints. HTML-Seiten leiten zur Login-Seite um, API-Calls
  bekommen 401. Logout: `GET /salons/{slug}/logout`.
- **Env-Keys** (alle optional, ohne sie ist der jeweilige Teil offen —
  nur für lokale Entwicklung okay):
  - `SESSION_SECRET`: signiert Sessions; ungesetzt = zufällig pro
    Prozessstart (Sessions sterben bei Neustart).
  - `FAMULOR_API_KEY`: wenn gesetzt, verlangen die
    `/famulor/tools/*`-Endpoints den Header `X-API-Key`.
  - `PLATFORM_ADMIN_KEY`: wenn gesetzt, verlangt `POST /salons` den
    Header `X-Admin-Key`.
- Keine Extra-Dependencies: PBKDF2 + HMAC aus der Stdlib.

## Deployment (docs/deploy.md)

Zwei fertige Wege, Anleitung in `docs/deploy.md`:

- **Vercel** (Wunsch des Nutzers): `api/index.py` exponiert die
  FastAPI-App als Serverless Function, `vercel.json` routet alles dorthin.
  **Zwingend nötig:** `SALON_DATABASE_URL` auf Postgres (Neon/Vercel
  Postgres) — Vercels Dateisystem ist ephemer, SQLite würde alle
  Buchungen verlieren. `postgres://`-URLs werden in `config.py`
  automatisch auf `postgresql://` (psycopg2) umgeschrieben. Außerdem
  `SESSION_SECRET` fest setzen (mehrere Serverless-Instanzen!). Die
  Root-`requirements.txt` enthält deshalb NUR das Backend; die Alt-App-
  Dependencies liegen in `requirements-hotel-app.txt`.
- **Docker** (Railway/Render/eigener Server): `Dockerfile`, SQLite auf
  Volume `/data`, PORT-Env wird respektiert.

Das eigentliche Hosten braucht einen Account beim Anbieter (bzw. ein
`VERCEL_TOKEN` in der Session) — ohne das kann eine KI-Session nur
vorbereiten, nicht deployen.

## Kalender-UI (`GET /salons/{slug}/calendar`)

Selbst-enthaltene HTML-Wochenansicht (Mo–So, 08–20 Uhr, Prev/Heute/Next),
Daten client-seitig von `GET /salons/{slug}/bookings?date_from&date_to`.
Stornierte Termine durchgestrichen; ×-Button storniert mit Rückfrage
(`DELETE /salons/{slug}/bookings/{id}`). Der Bot kann ebenfalls
stornieren (`cancel_appointment`: Startzeit + Telefonnummer oder Name).

## Aktueller Stand

- **52/52 Tests grün** (`python -m pytest tests/ -v`), komplett ohne
  externe Abhängigkeiten — die Engine ist unsere eigene
- UIs (Kalender + Admin + Login) im Browser verifiziert (Playwright)
- **Login pro Salon aktiv** (siehe Auth-Abschnitt); Famulor-Endpoints und
  Salon-Anlage per Env-Key absicherbar
- Deployment vorbereitet (Dockerfile + docs/deploy.md), aber noch nicht
  gehostet — braucht Railway/Render-Account des Nutzers
- Keine Anbindung an Famulor selbst (nur das Tool-Schema existiert)
- Kein Onboarding-Formular für neue Salons (nur `POST /salons`)

## Offene Punkte / Was als Nächstes gebraucht wird

1. **SMS-Anbieter** (Twilio, Seven, etc.) für Bestätigungs-/Reminder-SMS —
   vorher liefen SMS über Cal.com-Workflows, das ist mit dem Umbau
   entfallen. Nummern liegen normalisiert an jeder Buchung
   (`Booking.customer_phone`), ein Versand-Hook gehört am saubersten in
   `booking.create_booking`/`cancel_booking` + ein Scheduler für Reminder.
2. **Hosting durchführen**: Vercel-Setup + Dockerfile liegen bereit
   (docs/deploy.md) — es fehlt nur der Vercel-Login des Nutzers (Repo
   importieren, Neon-Postgres anlegen, Env-Vars setzen) oder ein
   `VERCEL_TOKEN`, mit dem eine KI-Session selbst deployen kann.
3. **Famulor-Account-Zugang** + Doku, wie dort Custom-Tools
   (Function-Calling) konfiguriert werden — `GET /famulor/tools` liefert
   das Schema; `X-API-Key`-Header mitkonfigurieren.
4. Echte Pilot-Salon-Daten (Mitarbeiter, Services, Dauern, Öffnungszeiten).
5. Später bei Multi-Prozess-Deployment: Doppelbuchungs-Schutz über
   DB-Constraint/Locking härten (aktuell: Re-Check in derselben
   Transaktion, reicht für einen Prozess).
6. Nice-to-have: Urlaubs-/Abwesenheitszeiten pro Mitarbeiter,
   Mittagspausen, Termin-Verschieben (statt Storno+Neubuchung).

## Wie testen

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Lokal starten:

```bash
uvicorn salon.main:app --reload
# dann z.B. POST /salons (siehe tests/test_api_e2e.py für ein Beispiel-Payload)
# Admin:    http://127.0.0.1:8000/salons/<slug>/admin
# Kalender: http://127.0.0.1:8000/salons/<slug>/calendar
```
