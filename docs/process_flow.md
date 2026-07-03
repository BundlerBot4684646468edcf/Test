# Prozess-Flow: Vom Anruf bis zum Termin

Kompletter Ablauf eines Termins, Ende-zu-Ende. Seit dem Umbau läuft
**alles in unserem eigenen Backend** — kein Cal.com mehr.

```
1. Kunde ruft an / schreibt Chat
        │
        ▼
2. Famulor (Voice/Chat AI) nimmt das Gespräch an
        │  Famulor erkennt Wunsch: Service + ggf. Mitarbeiter + Zeitraum
        ▼
3. Famulor ruft Tool "get_current_datetime" auf   ← Pflicht vor Datums-Deutung
        │  POST /famulor/tools/get_current_datetime
        │  → heutiges Datum, Uhrzeit, Wochentag (DE/EN), nächste 7 Tage
        ▼
4. Famulor ruft Tool "get_availability" auf
        │  POST /famulor/tools/get_availability
        │  { salon_slug, service_name, employee_name?, date_from, date_to }
        ▼
5. Unsere Buchungs-Engine (salon/booking.py)
        │  - Öffnungszeiten des Salons je Wochentag
        │  - Dauer: eigene Dauer des Mitarbeiters oder Service-Standard
        │  - Puffer (Aufräumzeit) nach jedem Termin
        │  - kein employee_name → Slot frei, wenn IRGENDEIN qualifizierter
        │    Mitarbeiter frei ist (Round-Robin)
        ▼
6. Famulor liest Kunde 2-3 Slots vor, Kunde wählt einen
        ▼
7. Famulor ruft Tool "book_appointment" auf
        │  { salon_slug, service_name, employee_name?, start_at,
        │    customer_name, customer_email, customer_phone? }
        ▼
8. Engine prüft NOCHMAL in derselben Transaktion:
        │  - Termin in der Zukunft? Innerhalb Öffnungszeiten?
        │  - Mitarbeiter (oder ein qualifizierter) wirklich noch frei?
        │    → Slot weg = HTTP 409 mit vorlesbarer Begründung
        │  - Round-Robin: am wenigsten ausgelasteter freier Mitarbeiter
        ▼
9. Buchung steht in unserer DB → Famulor bestätigt dem Kunden mündlich
        │  Telefonnummer wird E.164-normalisiert mitgespeichert
        ▼
10. Salon sieht den Termin sofort im eigenen Kalender
        │  GET /salons/{slug}/calendar (Wochenansicht, Storno per ×)
        ▼
11. Stornieren geht über beide Wege:
        - Bot: Tool "cancel_appointment" (Startzeit + Telefonnummer/Name)
        - Salon: ×-Button im Kalender (DELETE /salons/{slug}/bookings/{id})
```

## SMS (offen)

Bestätigungs-/Reminder-SMS liefen früher über Cal.com-Workflows und sind
mit dem Umbau entfallen. Für die Zukunft:

- Anbieter wählen (Twilio, Seven.io, …)
- Versand-Hook in `booking.create_booking` (Bestätigung) und
  `booking.cancel_booking` (Storno-Info)
- Reminder brauchen einen Scheduler (z. B. APScheduler/Cron), der
  Buchungen mit `start` in X Stunden abfragt und einmalig verschickt
- Die Nummer liegt bereits normalisiert an jeder Buchung
  (`Booking.customer_phone`)
