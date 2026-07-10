---
tags: [decision]
status: accepted
date: 2026-07-10
---

# Famulor als Voice-Plattform

## Kontext
Der KI-Rezeptionist braucht Telefonie + Voice-KI: Anrufe annehmen, natürlich sprechen, Transkripte liefern, später Termine buchen. Optionen: selbst bauen (Twilio + STT/LLM/TTS), oder eine fertige Voice-Agent-Plattform.

## Entscheidung
Wir bauen auf **Famulor** (app.famulor.de) auf.

## Begründung
- Assistenten, Telefonnummern, Transkripte und Anruf-Auswertung sind fertig — wir konzentrieren uns auf Prompt, Salon-Wissen und Vertrieb.
- Deutschsprachige Stimmen und Inbound-Handling funktionieren out of the box.
- Mid-Call-Tools ermöglichen später die Kalender-Anbindung für echte Terminbuchung.
- SMS/WhatsApp für Terminbestätigungen auf derselben Plattform.

## Konsequenzen
- Abhängigkeit von einem Anbieter (Preise, Verfügbarkeit) — Marge pro Salon hängt an Famulor-Kosten pro Anrufminute.
- Eigenbau bleibt als Option, falls Volumen und Anforderungen es später rechtfertigen — dann neuer Decision Record.
