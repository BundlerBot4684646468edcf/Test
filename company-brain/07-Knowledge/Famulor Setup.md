---
tags: [knowledge, engineering]
---

# Famulor Setup

Wie unser KI-Rezeptionist auf Famulor läuft und wie ein neuer Salon aufgesetzt wird.

## Zugang
- Plattform: https://app.famulor.de (Konto: alex.grunberer@gmail.com)
- Doku: https://docs.famulor.io

## Aktuelle Assistenten
| Assistent | ID | Zweck |
|---|---|---|
| `sfiumabiondei` | 16347 | Pilot-Salon, live mit eigener Nummer |
| `friseur test` | 15899 | Testassistent zum Ausprobieren von Prompts |

## Neuen Salon live schalten (Checkliste)
- [ ] Assistent anlegen (inbound), Sprache Deutsch, passende Stimme wählen
- [ ] System-Prompt aus Vorlage füllen: Salonname, Team, Leistungen + Preise, **Arbeitszeiten** (z. B. Di–Sa, 9:00–17:00), Ton („herzlich, per Du/Sie?")
- [ ] Telefonnummer zuweisen oder Rufumleitung von der Salonnummer einrichten
- [ ] Testanrufe machen: Öffnungszeiten fragen, Termin buchen, Sonderfall (Absage/Verschiebung)
- [ ] Salon-Notiz anlegen → [[06-Customers/Customers Index]] und verlinken

## Wöchentliche Qualitätsrunde
1. Anrufe der Woche durchgehen (Transkripte + Auswertung in Famulor)
2. Muster notieren: Wo versteht die KI etwas falsch? Was fragen Anrufer, das nicht im Prompt steht?
3. Prompt/Wissensbasis des Assistenten aktualisieren
4. Erkenntnisse hier oder in der Salon-Notiz festhalten

## Verwandt
- [[03-Product/Product Overview]] · [[08-Decisions/2026-07-10 Famulor als Voice-Plattform]]
