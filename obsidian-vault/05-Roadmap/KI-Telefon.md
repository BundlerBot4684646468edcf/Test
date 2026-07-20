---
tags: [mundpost, roadmap, feature, voice]
---

# KI-Telefonassistent (Säule 3)

Gehört zu: [[Mundpost]] · [[Roadmap]]

Nimmt Anrufe an, wenn niemand im Salon ans Telefon kann: Termine buchen/verschieben, Öffnungszeiten, Preise — 24/7, kein verpasster Anruf mehr. Das ist der Teil, den Agenturen als teuersten Baustein verkaufen.

## Warum es sich lohnt

- Friseursalons verlieren Termine, weil während der Arbeit niemand abhebt — jeder verpasste Anruf ist potenziell ein Kunde, der beim nächsten Salon anruft
- Der Bot kann beim Buchen direkt Name + Nummer erfassen → füttert automatisch die Kundenliste für Säule 1 (Bewertungen) und Säule 2 ([[Wallet-Stempelkarte]])
- Zweisprachig (Deutsch/Italienisch) ist für Südtirol ein echtes Verkaufsargument

## Zwei Umsetzungswege

### Weg A: Fertige Plattform — Famulor (Empfehlung für den Start)
- Sprachassistent wird auf der Plattform konfiguriert: Stimme, Sprache, System-Prompt, Telefonnummer
- Terminbuchung über "Mid-Call-Tools" (Webhook-Aufrufe an unser Railway-Backend oder direkt an einen Kalender)
- Famulor ist bereits als Connector im Claude-Konto verbunden (muss nur in den claude.ai-Connector-Einstellungen autorisiert werden, dann kann Claude den Bot direkt mitbauen)
- Kosten: pro Anrufminute, kein eigener Infrastrukturaufwand

### Weg B: Selbst bauen (Twilio Voice + Speech-APIs)
- Maximale Kontrolle, keine Plattform-Abhängigkeit
- Deutlich mehr Aufwand: Audio-Streaming, Speech-to-Text, Text-to-Speech, Gesprächslogik, Fehlerfälle
- Erst sinnvoll, wenn Weg A an Grenzen stößt

## Anbindung an Mundpost

1. Anruf → Bot bucht Termin → legt `Customer` (Name, Nummer, Termin als `servedAt`-Basis) direkt im Backend an
2. Nach dem Termin läuft automatisch die bestehende Bewertungs-Pipeline ([[Der komplette Ablauf]])
3. Optional: Bot erwähnt die Stempelkarte ("Übrigens, bei uns gibt's eine digitale Treuekarte…")

## Offene Fragen

- [ ] Welcher Kalender? (Google Calendar, Buchungssystem des Salons, eigenes Terminmodul im Backend?)
- [ ] Italienisch + Deutsch: eine Nummer mit Spracherkennung oder zwei Nummern?
- [ ] Wer zahlt die Anrufminuten — im Mundpost-Preis inklusive oder Weiterberechnung?

## Status

Idee/geplant — Start nach [[Wallet-Stempelkarte]]. Siehe Priorisierung in [[Roadmap]].
