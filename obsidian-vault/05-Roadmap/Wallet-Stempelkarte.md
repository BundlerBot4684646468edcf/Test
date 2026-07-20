---
tags: [mundpost, roadmap, feature, wallet]
---

# Wallet-Stempelkarte (Säule 2)

Gehört zu: [[Mundpost]] · [[Roadmap]]

Digitale 10-Punkte-Treuekarte in Apple Wallet / Google Wallet — mit Lock-Screen-Benachrichtigungen als eigenem Marketingkanal.

## Warum das stark ist

1. **Kontaktkanal ohne Telefonnummer**: Pass-Updates erzeugen Benachrichtigungen auf dem Sperrbildschirm — keine SMS-Kosten, keine Nummer nötig, kein Spam-Filter
2. **Wiederkehr-Anreiz**: 10 Stempel = Belohnung (z.B. gratis Haarschnitt) → Kunden kommen öfter
3. **Bewertungs-Moment**: Nach dem Stempel kommt die Notification ("Stempel 7/10 ✂️") — dort lässt sich die Bewertungsbitte einbauen (z.B. ab dem 2.–3. Besuch, wenn die Beziehung da ist)
4. Verzahnt sich mit Säule 1 ([[Foto-Personalisierung]]): Bewertungslink + Foto auf der Pass-Rückseite

## Ablauf aus Kundensicht

1. Erster Besuch: QR-Code an der Kasse scannen → Karte landet direkt in Apple/Google Wallet
2. Beim Bezahlen: André scannt den Barcode auf der Karte (oder tippt Kurzcode) → Stempel +1
3. Pass aktualisiert sich automatisch auf dem Handy, mit Lock-Screen-Notification
4. 10. Stempel → Belohnung + Glückwunsch-Notification

## Technik-Skizze

- **Apple Wallet**: `.pkpass`-Dateien, signiert mit Pass-Type-ID-Zertifikat; Updates via APNs-Push + Web-Service-Endpunkte (Registrierung/Update) — Node-Bibliothek: `passkit-generator`
- **Google Wallet**: Loyalty-Pass über die Google Wallet API (kostenloses API-Konto)
- **Backend**: läuft auf dem bestehenden Railway-Server; neue Tabellen `LoyaltyCard` (customerId, stampCount, passSerial) + Scan-Endpunkt; einfache Scan-Seite fürs Salon-Tablet/Handy
- **Verknüpfung**: Karte gehört zu einem `Customer` → Stempel-Event kann direkt eine Bewertungsanfrage triggern (statt CSV-Import)

## Voraussetzungen

- [ ] **Apple Developer Account, 99 €/Jahr** — Pflicht fürs Signieren; Registrierung dauert 1–2 Tage → früh beantragen!
- [ ] Google Wallet API Konto (kostenlos)
- [ ] Entscheidung: Belohnung bei 10 Stempeln (muss der Salon festlegen)

## Status

Geplant — Start nach Abschluss von Säule 1 (Foto + erste echte Versände). Siehe Priorisierung in [[Roadmap]].
