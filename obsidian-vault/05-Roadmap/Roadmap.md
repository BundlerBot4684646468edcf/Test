---
tags: [mundpost, roadmap]
---

# Roadmap / Was noch fehlt

Gehört zu: [[Mundpost]]

## Kurzfristig (bevor der erste echte Kunde live geht)

- [ ] Echtes Inhaberfoto hochladen + Schild-Position kalibrieren (siehe [[Foto-Personalisierung]])
- [ ] Twilio-Konto aufladen (raus aus Trial) ODER erstmal komplett auf E-Mail-Kanal setzen
- [ ] Cloudflare R2 einrichten, falls Foto-SMS gewünscht ist (siehe [[Cloudflare R2]])
- [ ] Erste echte Kundenliste (CSV) importieren und einen Testlauf mit 2-3 echten Kontakten fahren
- [ ] `dailyBatchLimit` pro Betrieb sinnvoll setzen (nicht zu aggressiv für den Start)

## Mittelfristig

- [ ] Visuelles Einstell-Tool im Dashboard für die Schild-Position (statt Koordinaten per Hand raten) — siehe Grenzen in [[Foto-Personalisierung]]
- [ ] Italienische/europäische Twilio-Absendernummer oder Alphanumeric Sender ID (siehe [[SMS Zustellprobleme Italien]])
- [ ] Automatischer Trigger für "Kunde wurde bedient" statt manuellem CSV-Import (z.B. Kassensystem-Anbindung, POS-Webhook)
- [ ] Rechtliches prüfen: DSGVO-konforme Einwilligung für Kontaktaufnahme, Double-Opt-in?, Impressum/Datenschutzerklärung für die Mails

## Längerfristig / Ideen

- [ ] Mehrsprachigkeit (aktuell nur Deutsch — Südtirol ist auch italienischsprachig)
- [ ] Mehrere Betriebe pro Login / Multi-Tenant-Verwaltung im Dashboard
- [ ] A/B-Testing von Nachrichtentexten (welcher Wortlaut bringt mehr Klicks?)
- [ ] WhatsApp als dritter Kanal (oft beliebter als SMS in Italien)
- [ ] Automatische Foto-Perspektivkorrektur, damit das Schild auch bei leicht schiefen Fotos gerade wirkt

## Bewusst nicht geplant / out of scope

- Kein eigenes Bewertungs-Widget außerhalb von Google (Ziel ist echte Google-Bewertungen, kein Trick-System)
- Kein Versuch, negative Bewertungen vorab herauszufiltern ("Review Gating") — das verstößt gegen Googles Richtlinien und ist bewusst nicht Teil des Produkts
