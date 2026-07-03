# Projekt: Multi-Tenant Friseur/Salon-Buchungs-SaaS

Dieses Dokument ist der Kontext-Handoff für jede KI-Session (oder jeden
Entwickler), die hier weiterarbeitet. Es fasst zusammen: was gebaut wird,
warum genau so, was schon existiert, und was als Nächstes fehlt.

## Ziel

Ein SaaS, mit dem mehrere Friseursalons (Mandanten) Termine für
verschiedene Mitarbeiter mit unterschiedlichen Service-Dauern anbieten
können. Kunden sollen sowohl per **KI-Telefon-/Chat-Rezeptionist (Famulor)**
als auch ggf. manuell durch Salon-Personal buchen können.

Zielgruppe/Umfang: **Multi-Tenant SaaS** (beliebig viele fremde Salons,
nicht nur ein Standort).

## Zentrale Architektur-Entscheidung: Cal.com als Buchungs-Engine

Wir bauen die Verfügbarkeits-/Kalenderlogik **nicht selbst**, sondern
nutzen **Cal.com** dafür. Begründung: Konfliktprüfung, Pufferzeiten,
Zeitzonen, Kalender-Sync sind dort fertig und gut getestet — das selbst zu
bauen ist wochenlange Arbeit und eine klassische Bug-Quelle
(Doppelbuchungen, Race Conditions).

```
Kunde ──(Anruf/Chat/WhatsApp)──> Famulor (Voice/Chat AI)
                                      │
                                      ▼  Function-Calling: get_availability / book_appointment
                                  Unser FastAPI-Backend (salon/)
                                      │
                                      ▼  Cal.com v2 API
                                  Cal.com  (1 Team pro Salon, unter UNSEREM eigenen Account)
                                      │
                                      ▼  Workflows (intern in Cal.com)
                              SMS-Reminder, No-Show-Follow-up
```

**Wichtig:** Automatisierungen (Erinnerungen, No-Show, Review-Anfragen)
laufen **in Cal.com selbst** über deren "Workflows"-Feature — wir bauen
dafür keine eigene Automatisierungs-Schicht. SMS läuft über Cal.coms
eingebautes Credit-System (kein eigener Twilio-Account nötig), erfordert
aber den Cal.com **Team-Plan** pro Salon (Free-Plan hat keine Workflows).

## Multi-Tenancy: Team pro Salon, nicht Managed Users

Jeder Salon = ein **Cal.com Team** unter unserem eigenen Cal.com-Account
(nicht ein separater Cal.com-Account pro Kunde). Mitarbeiter werden zwar
technisch als Cal.com-Nutzer eingeladen, müssen sich aber nie einloggen —
Famulor und unser Backend steuern alles über die API. Der Salon-Inhaber
sieht nie Cal.com direkt, nur unser eigenes Dashboard (noch nicht gebaut).

Cal.com bietet zusätzlich ein "Managed Users"/"Platform"-Feature für noch
saubereres White-Labeling (isolierte Schatten-Accounts pro Kunde, kein
Cal.com-Branding sichtbar). **Unbestätigtes Rechercheergebnis:** Cal.com
scheint dieses Platform-Angebot seit Dez. 2025 nicht mehr für neue
Signups offen zu haben (auf Partnerschaftsmodell umgestellt) — das wurde
**nicht verifiziert** (Cal.com blockt automatisiertes Abrufen ihrer
Doku-Seiten). Falls später volles White-Labeling gebraucht wird: zuerst
direkt mit Cal.com-Sales klären, ob/wie das noch geht, bevor Aufwand
reingesteckt wird. Für jetzt: Team-pro-Salon-Lösung reicht und ist bereits
gebaut.

## Der Kniff bei unterschiedlichen Dauern pro Mitarbeiter

Cal.com speichert die Termin-Dauer **am Event-Type, nicht am Host**. Ein
einzelner Event-Type kann also nicht "Lena 45 Min, Tom 90 Min" für
denselben Service gleichzeitig abbilden. Deshalb legt das Onboarding pro
Service **zwei Arten von Event-Types** an:

1. **Ein Round-Robin-Team-Event-Type** mit allen qualifizierten
   Mitarbeitern als Hosts, Standarddauer des Service → genutzt, wenn der
   Kunde keinen Mitarbeiter-Wunsch hat ("egal wer").
2. **Pro (Mitarbeiter, Service) ein eigener Event-Type mit fixem Host**,
   dessen Dauer den Service-Standard überschreiben kann → genutzt, wenn
   ein Mitarbeiter namentlich gewünscht wird.

Famulor übergibt einfach `employee_name` (oder lässt es weg); unser
Backend löst daraus den richtigen `cal_event_type_id` auf
(`salon/famulor_tools.py::_resolve_event_type`). Die Slot-Länge ist dann
automatisch korrekt — kein Laufzeit-Rechnen nötig.

## Codebase-Struktur

```
salon/
  config.py        Env-Vars: CAL_API_KEY, CAL_API_BASE, SALON_DATABASE_URL
  db.py             SQLAlchemy Engine/Session (SQLite lokal, austauschbar gegen Postgres)
  models.py         Salon, Employee, Service, EmployeeService (Qualifikation+eigene Dauer+Event-Type)
  schemas.py        Pydantic-Schemas für die API (SalonOnboardIn, AvailabilityQuery, BookingIn, ...)
  cal_client.py      Wrapper um die Cal.com v2 API (Team/Host/Event-Type anlegen, Slots, Buchung, Buchungsliste)
  onboarding.py      onboard_salon(): provisioniert einen neuen Salon komplett in Cal.com
  famulor_tools.py   FAMULOR_TOOLS Schema (Function-Calling) + Handler, die Namen auf Cal.com-IDs auflösen
  calendar_ui.py     Read-only Wochen-Kalender (HTML) + Buchungs-Normalisierung für die Admin-Ansicht
  main.py            FastAPI-App: POST /salons, GET /famulor/tools,
                      POST /famulor/tools/get_current_datetime,
                      POST /famulor/tools/get_availability, POST /famulor/tools/book_appointment,
                      GET /salons/{slug}/bookings, GET /salons/{slug}/calendar

tests/
  test_salon_onboarding.py   Unit-Tests gegen FakeCalClient (kein Netzwerk nötig)
  test_api_e2e.py             E2E-Tests über die echten HTTP-Endpoints (FastAPI TestClient + gemockter Cal.com)

app.py              UNVERWANDTE Alt-App (Hotel-Reputation-MVP) — nicht anfassen, nicht Teil dieses Projekts
```

`app.py` und `requirements.txt`-Einträge für Streamlit/Plotly/etc. gehören
zu einem völlig anderen, älteren Projekt in diesem Repo (Hotel-Bewertungs-
Analyse) und haben mit der Salon-Buchung nichts zu tun.

## Datums-Sicherheit ("Samstag war gestern"-Bug, gefixt)

Der Voice-Bot kannte das aktuelle Datum nicht und hat Wochentage/relative
Daten ("morgen", "Samstag") falsch aufgelöst. Dagegen gibt es jetzt drei
Schichten:

1. **Tool `get_current_datetime`** (laut Tool-Beschreibung vor jeder
   Datumsinterpretation aufzurufen): heutiges Datum, Uhrzeit, Wochentag
   (DE/EN) in der Salon-Zeitzone plus die Wochentage der nächsten 7 Tage.
2. **`get_availability`**: Zeitraum komplett in der Vergangenheit → Fehler
   mit Klartext-Hinweis, welcher Tag heute ist; nur Start in der
   Vergangenheit → wird auf heute geklemmt. Antwort enthält immer
   `today`/`today_weekday_de`.
3. **`book_appointment`**: Termin-Start in der Vergangenheit → Fehler mit
   demselben Hinweis, Buchung erreicht Cal.com nie.

## SMS-Fix

SMS (Bestätigung/Reminder/Review über Cal.com Workflows) gingen nie raus,
weil `handle_book_appointment` die `customer_phone` verworfen hat — sie
kam nie bei Cal.com an. Jetzt wird sie best-effort auf E.164 normalisiert
(`0176…` → `+49176…`, Zielmarkt DE) und als `attendee.phoneNumber` in die
Buchung geschrieben. `customer_phone` ist im Famulor-Tool-Schema jetzt
required, damit der Bot die Nummer aktiv erfragt; das Backend selbst
bleibt tolerant und bucht auch ohne Nummer (dann ohne SMS).

## Kalender-UI

`GET /salons/{slug}/calendar` liefert eine selbst-enthaltene HTML-
Wochenansicht (Mo–So, 08–20 Uhr, Prev/Heute/Next-Navigation, stornierte
Termine durchgestrichen). Die Seite holt die Daten client-seitig von
`GET /salons/{slug}/bookings?date_from=…&date_to=…`, das alle Cal.com-
Buchungen über sämtliche Event-Types des Salons einsammelt
(`calendar_ui.py`) und die Zeiten in die Salon-Zeitzone konvertiert.
Read-only — gebucht/storniert wird weiterhin über Famulor bzw. Cal.com.

## Aktueller Stand

- Vollständiges Skeleton steht, **20/20 Tests grün** (`python -m pytest tests/ -v`)
- Alles bisher nur gegen einen **gemockten** Cal.com-Client getestet — **noch
  nie gegen die echte Cal.com-API verifiziert**, da kein API-Key vorhanden war
- `cal_client.py`-Endpunkt-Pfade/Payload-Felder (`lengthInMinutes`,
  `schedulingType`, `hosts[].isFixed` etc.) basieren auf Cal.coms
  dokumentierter v2-API, aber **nicht live verifiziert** — unbedingt gegen
  echten Account/echte API-Doku gegenchecken, sobald Zugriff da ist
- Kein Hosting/Deployment, läuft nur lokal
- Keine Anbindung an Famulor selbst (nur das Tool-Schema dafür existiert)
- Admin-Kalender (Wochenansicht) existiert unter `/salons/{slug}/calendar`;
  ein volles Dashboard (Onboarding-Formulare, Mitarbeiterverwaltung) fehlt weiter

## Offene Punkte / Was als Nächstes gebraucht wird

1. **Cal.com Team-Plan-Account + `CAL_API_KEY`** — kommt laut Nutzer "heute
   Abend" zusammen mit weiteren Zugängen
2. Echten API-Call gegen Cal.com machen und `cal_client.py` ggf. an die
   tatsächliche Payload-Struktur anpassen
3. **Hosting-URL** für das FastAPI-Backend (Railway/Render/Fly.io/eigener
   Server) — muss von Famulor und Cal.com-Webhooks von außen erreichbar sein
4. **Famulor-Account-Zugang** + Doku/Screenshots, wie dort Custom-Tools
   (Function-Calling) konfiguriert werden — bisher unbekannt, wie genau
   Famulor `FAMULOR_TOOLS` einbinden will
5. Echte Pilot-Salon-Daten (Mitarbeiter, Services, Qualifikationen/Dauern)
6. Klären: was passiert bei Konflikt, wenn ein Slot zwischen Verfügbarkeits-
   Abfrage und Buchung weg ist (Cal.com sollte das beim `create_booking`
   selbst ablehnen — verifizieren, sobald live getestet werden kann)
7. Falls später gebraucht: Cal.com-Sales fragen, ob Managed Users/Platform
   für neue Kunden noch verfügbar ist (siehe oben)

## Wie testen

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Alle Tests laufen ohne externe Abhängigkeiten (Cal.com wird durch
`FakeCalClient` in `tests/test_salon_onboarding.py` ersetzt).
