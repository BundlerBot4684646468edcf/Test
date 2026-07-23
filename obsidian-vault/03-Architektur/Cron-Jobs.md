---
tags: [mundpost, architektur, cron]
---

# Cron-Jobs

Gehört zu: [[Mundpost]] · [[Architektur Überblick]] · [[Der komplette Ablauf]]

Definiert in `src/services/cronJobs.ts`, läuft im selben Node-Prozess wie der Server (kein separater Worker).

| Zeitplan | Funktion | Zweck |
|---|---|---|
| `*/15 * * * *` (alle 15 Min) | `processReviewQueue()` (`services/reviewQueue.ts`) | fällige Anfragen verarbeiten: Opt-out-/Limit-Check, Foto personalisieren, SMS/E-Mail senden, Follow-up nach 24h (Kanalwechsel zu E-Mail) |
| `0 2 * * *` (täglich 2 Uhr) | `updateReviewMetrics()` (`services/reviewMetrics.ts`) | aktuelle Google-Sternebewertung abfragen, `ReviewEvent` speichern |

## Wichtig für lokales Testen

Der stündliche Rhythmus ist für Entwicklung unpraktisch — man will nicht bis zur nächsten vollen Stunde warten. Deshalb existieren die `/api/setup/test-*`-Endpunkte (siehe [[Test-Checkliste]]), die den Versand **sofort** auslösen und die Warteschlange umgehen — nur zum manuellen Testen, nicht Teil des eigentlichen Kundenflows.

## Serverneustart

Cron-Jobs laufen nur, solange der Node-Prozess läuft. Kein externer Scheduler — bei einem Serverneustart (z.B. Absturz) verpasste Läufe werden **nicht** nachgeholt, sondern einfach beim nächsten regulären Tick wieder aufgenommen (fällige Einträge bleiben in der DB stehen, bis sie verarbeitet werden).
