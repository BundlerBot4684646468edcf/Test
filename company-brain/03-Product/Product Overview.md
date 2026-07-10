---
tags: [product]
status: Pilot
---

# Produkt — amstudio KI-Rezeptionist

> Anruf im Salon → KI nimmt ab → beantwortet Fragen & bucht Termine → Salon bekommt Zusammenfassung.

## Was das Produkt macht
1. **Anruf annehmen** — die Salonnummer (oder eine neue Nummer) leitet auf den Famulor-Assistenten
2. **Natürlich sprechen** — Begrüßung im Ton des Salons, Deutsch, versteht Dialekt-Anrufer
3. **FAQ beantworten** — Öffnungszeiten (z. B. Di–Sa, 9:00–17:00), Preise, Leistungen, Anfahrt
4. **Termin buchen** — Wunschleistung + Zeit erfragen, freien Slot vorschlagen, buchen
5. **Nachbereiten** — Transkript & Auswertung pro Anruf; SMS/WhatsApp-Bestätigung an den Kunden möglich

## Live-Assistenten (Famulor)
| Assistent | ID | Typ | Status |
|---|---|---|---|
| sfiumabiondei | 16347 | Inbound, eigene Nummer | 🟢 live (Pilot-Salon) |
| friseur test | 15899 | Inbound, ohne Nummer | 🧪 Test/Spielwiese |

Details & Konfiguration: [[07-Knowledge/Famulor Setup]]

## Stack
- **Voice-Plattform:** Famulor (app.famulor.de) — Assistenten, Nummern, Transkripte, Mid-Call-Tools
- **Kanäle:** Telefon (inbound), optional SMS/WhatsApp-Bestätigungen
- Entscheidung dazu: [[08-Decisions/2026-07-10 Famulor als Voice-Plattform]]

## Roadmap
| Horizont | Thema | Notizen |
|---|---|---|
| Jetzt | Pilot sfiumabiondei stabil: Transkripte wöchentlich prüfen, Prompt schärfen | [[05-Projects/Pilot Program]] |
| Als Nächstes | Terminbuchung an Salon-Kalender anbinden (Mid-Call-Tool) | Kalender-Entscheidung nötig → [[08-Decisions/Decisions Index]] |
| Später | Onboarding-Vorlage: neuen Salon in < 1 Tag live schalten | |

## Verwandt
- [[06-Customers/Customers Index]] · [[07-Knowledge/Famulor Setup]]
