---
tags: [mundpost, architektur, datenbank]
---

# Datenmodell

Gehört zu: [[Mundpost]] · [[Architektur Überblick]]

SQLite-Datei: `mundpost.db` (ein einzelnes File, per `node:sqlite`). Schema + Migrationen in `src/db.ts`.

## Business (ein Betrieb, z.B. ein Salon)

| Feld | Typ | Bedeutung |
|---|---|---|
| id | TEXT (PK) | |
| name | TEXT | Betriebsname |
| ownerName | TEXT | Name des Inhabers (für Nachrichtentext: "Eine kurze Frage von Marco") |
| ownerPhotoUrl | TEXT | URL zum Inhaberfoto mit leerem Schild |
| googlePlaceId / googleReviewLink | TEXT | von Google Places API befüllt |
| timezone | TEXT | Default `Europe/Rome` |
| dailyBatchLimit | INTEGER | max. Anfragen/Tag (Default 20) |
| signX, signY, signWidth, signHeight, signRotation | REAL | Schild-Position im Foto, siehe [[Foto-Personalisierung]] (nachträglich per Migration ergänzt) |

## Customer (Endkunde des Betriebs)

| Feld | Typ | Bedeutung |
|---|---|---|
| id | TEXT (PK) | |
| businessId | TEXT (FK) | |
| firstName | TEXT | wird aufs Schild geschrieben + in Nachrichten verwendet |
| phone / email | TEXT | mindestens eins muss gesetzt sein |
| servedAt | TEXT | wann der Kunde bedient wurde |
| source | TEXT | `past` (Altkunden-Import) oder `new` (laufender Zulauf) |
| optOut | INTEGER (0/1) | Kunde hat sich dauerhaft abgemeldet |

## ReviewRequest (eine geplante/gesendete Bewertungsanfrage)

| Feld | Typ | Bedeutung |
|---|---|---|
| id | TEXT (PK) | |
| customerId, businessId | TEXT (FK) | |
| channel | TEXT | `sms` oder `email` |
| status | TEXT | `queued` → `sent` → `reminded` (oder `opted_out`) |
| scheduledAt, sentAt, remindedAt, reviewedAt | TEXT | Zeitstempel je Phase |

Status-Übergänge siehe [[Der komplette Ablauf]].

## ReviewEvent (Snapshot der Google-Bewertung über Zeit)

| Feld | Typ | Bedeutung |
|---|---|---|
| id | TEXT (PK) | |
| businessId | TEXT (FK) | |
| newReviewCount | INTEGER | neue Bewertungen seit letztem Check |
| avgRating | REAL | aktuelle Ø-Sternebewertung |
| totalReviews | INTEGER | Gesamtzahl Bewertungen |
| checkedAt | TEXT | wann abgefragt (täglich, 2 Uhr) |

Wird vom Dashboard-Chart genutzt, um den Bewertungsverlauf darzustellen.

## Beziehungen

```
Business 1---* Customer 1---* ReviewRequest
Business 1---* ReviewEvent
```

Alle Fremdschlüssel mit `ON DELETE CASCADE` — löscht man einen Betrieb, verschwinden auch seine Kunden/Anfragen/Events automatisch.

## Migrations-Ansatz

Kein Migrationstool — `ensureColumn()` in `db.ts` prüft bei jedem Serverstart per `PRAGMA table_info`, ob eine Spalte schon existiert, und fügt sie sonst per `ALTER TABLE` hinzu. Einfach, aber nur für additive Änderungen geeignet (neue Spalten) — kein Spalten-Umbenennen/Löschen.
