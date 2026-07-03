# Prozess-Flow: Vom Anruf bis zur Review-SMS

Kompletter Ablauf eines Termins, Ende-zu-Ende, inkl. welcher Teil wo läuft
(unser Code vs. Cal.com). Cal.com-Workflow-Schritte (Reminder, Review) sind
**Konfiguration in der Cal.com-UI**, kein Code in diesem Repo — siehe unten
für die exakten Einstellungen.

```
1. Kunde ruft an / schreibt Chat
        │
        ▼
2. Famulor (Voice/Chat AI) nimmt das Gespräch an
        │  Famulor erkennt Wunsch: Service + ggf. Mitarbeiter + Zeitraum
        ▼
3. Famulor ruft Tool "get_availability" auf
        │  POST /famulor/tools/get_availability
        │  { salon_slug, service_name, employee_name?, date_from, date_to }
        ▼
4. Unser Backend (salon/famulor_tools.py::handle_get_availability)
        │  - Salon per slug finden
        │  - Service per Name finden (case-insensitive)
        │  - Event-Type auflösen (_resolve_event_type):
        │      kein employee_name  -> Round-Robin "any" Event-Type
        │      employee_name       -> fixer Event-Type dieses Mitarbeiters
        │                             (eigene Dauer!)
        ▼
5. Cal.com API: GET /slots (cal_client.py::get_slots)
        │  liefert freie Slots für genau diesen Event-Type
        ▼
6. Famulor liest Kunde 2-3 Slots vor, Kunde wählt einen
        ▼
7. Famulor ruft Tool "book_appointment" auf
        │  POST /famulor/tools/book_appointment
        │  { salon_slug, service_name, employee_name?, start_at,
        │    customer_name, customer_email, customer_phone? }
        ▼
8. Unser Backend (handle_book_appointment)
        │  - gleiche Auflösung wie oben (Salon, Service, Event-Type)
        ▼
9. Cal.com API: POST /bookings (cal_client.py::create_booking)
        │  Cal.com prüft selbst auf Konflikte/Doppelbuchung und legt
        │  den Termin im Kalender des Mitarbeiters an
        ▼
10. Booking steht in Cal.com → Famulor bestätigt dem Kunden mündlich
        │
        ▼  ──────────── Ab hier: alles in Cal.com selbst, kein Code ────────────
        │
11. Cal.com Workflow "Booking Confirmation" feuert sofort
        │  Trigger: "When booking is created" → SMS + E-Mail an Kunde
        ▼
12. Cal.com Workflow "Reminder" feuert X Stunden vor dem Termin
        │  Trigger: "Before the event starts", Offset z.B. 24h → SMS
        ▼
13. Termin findet statt (oder No-Show)
        ▼
14. Cal.com Workflow "Review Request" feuert nach Termin-Ende
        │  Trigger: "After the event ends", Offset z.B. +2h → SMS mit
        │  Bewertungslink (Google/Trustpilot/eigene Review-Seite)
        ▼
15. (optional) Cal.com Workflow "No-Show Follow-up"
        │  Trigger: "After the event ends" + Bedingung "no-show" →
        │  SMS mit Rebuchungs-Link
```

## Was läuft wo

| Schritt | System | Code/Config |
|---|---|---|
| Gespräch führen, Absicht erkennen | Famulor | Famulor-eigene Konfiguration (Prompt + Tools) |
| Verfügbarkeit/Buchung ausführen | unser FastAPI-Backend | `salon/famulor_tools.py`, `salon/cal_client.py` |
| Konfliktprüfung, Kalender, Zeitzonen | Cal.com | Cal.com selbst (verifiziert sobald API-Key da ist) |
| Booking-Bestätigung, Reminder, Review, No-Show | Cal.com Workflows | **reine UI-Konfiguration in Cal.com**, kein Code hier |

## Cal.com Workflows einrichten: Booking-Bestätigung + Review-SMS gleichzeitig

Beide Workflows sind voneinander unabhängig und können parallel auf
denselben Event-Types aktiv sein — sie haben unterschiedliche Trigger, es
gibt keinen Konflikt. So werden sie pro Salon (= Cal.com Team) angelegt:

**Voraussetzung:** Team-Plan für das jeweilige Salon-Team (Free-Plan hat
keine Workflows) + SMS-Guthaben/-Credits in Cal.com.

**Voraussetzung im Backend:** Die Kundennummer muss als
`attendee.phoneNumber` in der Buchung ankommen — ohne sie verschickt
Cal.com für diese Buchung keine SMS. `book_appointment` reicht
`customer_phone` jetzt E.164-normalisiert (`0176…` → `+49176…`) an Cal.com
durch (vorher wurde sie verworfen); die Tool-Beschreibung weist Famulor an,
die Anrufer-Nummer zu nutzen oder sonst danach zu fragen.

### Workflow 1 — Booking Confirmation (sofort)
1. Cal.com → Team des Salons → **Workflows** → "+ New Workflow"
2. **Trigger**: "When booking is created"
3. **Event Types**: alle Event-Types dieses Salons zuweisen (oder "All")
4. **Action**: "Send SMS to attendee"
5. **Message**: z.B.
   `Hallo {ATTENDEE}, dein Termin bei {ORGANIZER} am {EVENT_DATE} um {EVENT_TIME} ist bestätigt.`
6. Speichern → aktiv

### Workflow 2 — Review Request (nach Termin-Ende)
1. Gleicher Team → Workflows → "+ New Workflow"
2. **Trigger**: "After the event ends"
3. **Offset**: z.B. +2 Stunden (oder am nächsten Morgen, je nach Wunsch)
4. **Event Types**: gleiche Zuweisung wie oben
5. **Action**: "Send SMS to attendee"
6. **Message**: z.B.
   `Danke für deinen Besuch bei {ORGANIZER}! Wie war's? Bewerte uns hier: <Review-Link>`
7. Speichern → aktiv

Beide Workflows laufen ab dann automatisch und unabhängig für jede neue
Buchung auf den zugewiesenen Event-Types — kein zusätzlicher Code in
diesem Repo nötig, kein Webhook-Handling auf unserer Seite erforderlich.

### Optional: Reminder + No-Show-Follow-up
Gleiches Prinzip, zusätzliche Workflows mit Trigger "Before the event
starts" (Reminder) bzw. "After the event ends" + Filter auf No-Show
(Follow-up mit Rebuchungslink).

## Noch nicht verifiziert (live testen sobald Cal.com-Zugang da ist)

- Exakte Trigger-/Offset-Optionen und Platzhalter-Namen (`{ATTENDEE}` etc.)
  in der aktuellen Cal.com-Version
- Ob SMS-Versand pro Team gesondert aktiviert/freigeschaltet werden muss
- Tatsächliches Timing (Cal.com-interne Queue-Verzögerung bei "sofort"-Trigger)
- Verhalten bei `employee_name`-spezifischen Event-Types: laufen Workflows
  pro Event-Type oder global pro Team? (Doku legt nahe: pro Event-Type-
  Zuweisung, aber unverifiziert)

Unser Backend (`salon/`) muss für diesen Teil **nichts** zusätzlich tun —
es endet mit Schritt 9 (Booking erstellen). Alles danach ist Cal.com-
Workflow-Konfiguration, kein Python-Code.
