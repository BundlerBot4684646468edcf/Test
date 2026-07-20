---
tags: [mundpost, overview]
---

# Mundpost 🗣️📮

Google-Bewertungs-Automatisierung für Südtiroler Betriebe. Nach jedem Kundenbesuch schickt Mundpost automatisch eine freundliche SMS/E-Mail mit Bitte um eine Google-Bewertung — inklusive eines **personalisierten Fotos** vom Inhaber, der ein Schild mit dem handgeschriebenen Namen des Kunden hochhält (das virale TikTok-Format).

## Schnellzugriff

- Setup von Null: [[Lokales Setup]]
- API-Keys besorgen: [[API Keys besorgen]]
- Alles durchtesten: [[Test-Checkliste]]
- Wie ein Kunde eingerichtet wird: [[Neuen Kunden anlegen]]
- Der komplette Ablauf: [[Der komplette Ablauf]]
- Technischer Aufbau: [[Architektur Überblick]]
- Datenmodell: [[Datenmodell]]
- Foto-Personalisierung im Detail: [[Foto-Personalisierung]]
- Bekannte Probleme & Fixes: [[Troubleshooting]]
- Online hosten (Railway + Vercel): [[Railway Hosting]]
- Was noch fehlt: [[Roadmap]]
- Nächste Säulen: [[Wallet-Stempelkarte]] · [[KI-Telefon]]

## Ein-Satz-Zusammenfassung

> Kunde besucht Betrieb → Mundpost merkt sich ihn → nach ein paar Stunden/Tagen automatische SMS/E-Mail mit personalisiertem Foto + Link zur Google-Bewertung → Kunde bewertet → Mundpost trackt Sternebewertung über Zeit.

## Projekt-Ordner

- Backend: `mundpostworking/` (Node + Express + SQLite, Port 3000)
- Frontend: `mundpostworking/frontend/` (Next.js Dashboard, Port 3001)
- Uploads (Fotos): `mundpostworking/uploads/`
- Datenbank-Datei: `mundpostworking/mundpost.db` (SQLite, eine einzelne Datei)

## Stack auf einen Blick

| Bereich | Technologie | Warum |
|---|---|---|
| Backend | Node.js + Express + TypeScript | einfach, schnell |
| DB | SQLite (`node:sqlite`, in Node 22 eingebaut) | kein Server, keine Docker, kein Netzwerk nötig |
| SMS | Twilio | Standard, gute Doku |
| E-Mail | Resend | einfache API, zuverlässige Zustellung |
| Foto-Rendering | `@napi-rs/canvas` + Caveat-Font | Name wird handschriftlich aufs Foto gemalt |
| Foto-Speicher | lokal (`/uploads`) im Dev, Cloudflare R2 in Produktion | MMS braucht eine öffentliche URL |
| Frontend | Next.js + Recharts | Dashboard mit Statistiken |

## Status (Stand 18.07.2026)

- ✅ Backend läuft lokal, SQLite funktioniert offline
- ✅ SMS-Versand getestet (Twilio Trial, verifizierte Nummer)
- ✅ E-Mail-Versand getestet (Resend)
- ✅ Dashboard mit Redesign fertig
- ✅ Foto-Personalisierung ("Name aufs Schild") fertig und getestet
- ⏳ Cloudflare R2 noch nicht eingerichtet (nötig für Foto-**MMS**, E-Mail mit Foto geht schon)
- ⏳ Noch kein echtes Kundenfoto hochgeladen / Schild-Position noch nicht kalibriert
- ⏳ Twilio noch im Trial-Modus (nur verifizierte Nummern, 5 SMS/Tag Limit)
