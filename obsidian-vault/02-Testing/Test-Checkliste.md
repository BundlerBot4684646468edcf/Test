---
tags: [mundpost, testing]
---

# Test-Checkliste

Gehört zu: [[Mundpost]] · [[Lokales Setup]]

Alles zum Copy-Pasten in den Browser (GET-Endpunkte) oder mit `curl`/Postman (POST/PUT). Backend muss auf Port 3000 laufen.

## 1. Health & Status

```
GET http://localhost:3000/health
GET http://localhost:3000/api/setup/status
```
→ zeigt, ob Google Places / SMS / E-Mail konfiguriert sind.

## 2. SMS testen

```
GET http://localhost:3000/api/setup/test-sms/393273042753
```
→ sendet Test-SMS an die Nummer (ohne `+`, ohne Leerzeichen, im Pfad).

**Bei Trial-Konto:** Nummer muss vorher in der Twilio-Konsole verifiziert sein, sonst Fehler `21608`. Max. 5/Tag (`63038`).

Zustellstatus direkt mitprüfen (wartet bis zu ~18s auf finalen Status):
```
GET http://localhost:3000/api/setup/test-sms-check/393273042753
```

Status eines bereits verschickten SMS nachträglich prüfen:
```
GET http://localhost:3000/api/setup/sms-status/:sid
```
(`:sid` aus der Antwort eines vorherigen Sends)

## 3. E-Mail testen

```
GET http://localhost:3000/api/setup/test-email?to=deine@email.de
```
→ einfache Test-Mail ohne Foto.

```
GET http://localhost:3000/api/setup/test-email-photo/deine@email.de
```
→ volle Vorschau-Mail **mit Foto** (nimmt ein öffentliches Beispielfoto, wenn keins hochgeladen ist). Eigenes Foto testen mit `?photo=https://...`.

## 4. Foto-Personalisierung testen ("Name aufs Schild")

**Ohne eigenes Foto** (Demo-Rotwand + Schild):
```
GET http://localhost:3000/api/setup/test-photo/Anna
GET http://localhost:3000/api/setup/test-photo/Marco
```
→ zeigt direkt im Browser das PNG. Muss zeigen: roter Hintergrund, weißes Schild, Name in Handschrift.

**Mit echtem Betriebsfoto** — siehe [[Foto-Personalisierung]] für die vollen Schritte:
1. Foto hochladen (`POST /api/businesses/:id/photo`)
2. Schild-Position einstellen (`PUT /api/businesses/:id/photo/sign-box`)
3. Vorschau: `GET /api/businesses/:id/photo/personalized?name=Anna`

## 5. MMS testen (nur mit R2 sinnvoll)

```
GET http://localhost:3000/api/setup/test-mms/393273042753
```
→ testet, ob überhaupt Bild-SMS beim Handy ankommt (Twilio-Demo-Bild). Auf Trial-Konten oft unzuverlässig nach Italien.

## 6. Business + Kunden anlegen (End-to-End)

Siehe [[Neuen Kunden anlegen]] und [[Der komplette Ablauf]] für den ganzen Weg von "Betrieb anlegen" bis "Bewertungsanfrage kommt an".

## Bekannte Fehlermeldungen

| Fehler | Bedeutung | Fix |
|---|---|---|
| `Error: Cannot find module '@fontsource/...'` | `npm install` nach Update nicht/nicht vollständig gelaufen | `npm install` erneut ausführen, Server neu starten |
| SMS-Fehler `21608` | Trial-Konto, Zielnummer nicht verifiziert | Nummer in Twilio-Konsole verifizieren |
| SMS-Fehler `63038` | Trial-Limit 5 SMS/Tag erreicht | warten oder Konto aufladen |
| SMS-Fehler `30008` / `undelivered` | US-Absendernummer → Zustellung nach Italien unzuverlässig | siehe [[SMS Zustellprobleme Italien]] |
| Foto zeigt Kästchen (□□□) statt Text | Schriftart im Canvas nicht registriert | betrifft nur alte Version vor dem Inter-Font-Fix — sollte erledigt sein |
| Dashboard zeigt alte Version | Alter Serverprozess blockiert den Port noch | siehe [[Troubleshooting]] |

Vollständige Liste: [[Troubleshooting]]
