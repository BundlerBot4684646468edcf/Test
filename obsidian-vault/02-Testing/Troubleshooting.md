---
tags: [mundpost, testing, troubleshooting]
---

# Troubleshooting

Gehört zu: [[Mundpost]] · [[Test-Checkliste]]

## Dashboard zeigt alte Version / "sieht gleich aus"

**Ursache:** Ein alter Server-Prozess blockiert noch Port 3000 oder 3001, während ein neuer daneben läuft.

**Fix (Windows):**
1. Alle laufenden Node-Prozesse beenden:
   ```
   taskkill /F /IM node.exe
   ```
2. Neu starten — **erst Backend, dann Frontend**:
   ```
   cd mundpostworking && npm run dev
   ```
   (neues Terminal-Fenster)
   ```
   cd mundpostworking/frontend && npm run dev
   ```
3. Browser-Hardreload (Strg+F5), da auch der Browser-Cache alte Assets halten kann.

## `Error: Cannot find module '@fontsource/...'` o.ä. nach ZIP-Update

**Ursache:** `package.json` wurde ersetzt (neue Abhängigkeit gelistet), aber `npm install` wurde nicht (erneut) ausgeführt — `node_modules` ist noch auf altem Stand.

**Fix:**
```
cd mundpostworking
npm install
npm run dev
```

## ZIP entpackt sich in einen eigenen Unterordner statt direkt in `mundpostworking`

Normales Verhalten von Windows beim Entpacken. Inhalt (`src`, `package.json`, `package-lock.json`) manuell nach `mundpostworking` kopieren und **ersetzen** lassen — nicht den entpackten Ordner umbenennen. Details: [[Lokales Setup]].

## SMS kommt nicht an

Reihenfolge zum Prüfen:
1. `http://localhost:3000/api/setup/status` → ist `sms.configured: true`?
2. Trial-Konto? → Zielnummer in Twilio-Konsole verifiziert? (sonst Fehler `21608`)
3. Tageslimit 5 SMS erreicht? (Fehler `63038`) → siehe [[API Keys besorgen]]
4. Zustellstatus abfragen: `/api/setup/test-sms-check/:number` oder `/api/setup/sms-status/:sid`
5. Status `undelivered` + US-Nummer als Absender → siehe [[SMS Zustellprobleme Italien]]

## Foto zeigt Kästchen/Tofu-Boxen (□□□) statt Text

War ein Bug bis 18.07.2026: Der Demo-Header-Text nutzte `sans-serif`, das `@napi-rs/canvas` auf manchen Windows-Systemen nicht auflösen konnte (keine Systemschrift gefunden). **Fix bereits eingespielt** — `@fontsource/inter` wird jetzt zusätzlich zur Handschrift gebündelt. Falls es doch nochmal auftaucht: prüfen, ob `@fontsource/inter` in `package.json` steht und `npm install` gelaufen ist.

## Twilio MMS (Foto per SMS) kommt nicht an / unzuverlässig

Bekannte Einschränkung, kein Bug: MMS nach Italien ist mit US-Absendernummern und/oder Trial-Konten stark eingeschränkt. **E-Mail mit eingebettetem Foto ist der zuverlässigere Kanal** und bereits so implementiert — für Produktion siehe [[SMS Zustellprobleme Italien]].

## "Kein Foto hochgeladen"-Fehler bei `/photo/personalized`

Erst `POST /api/businesses/:id/photo` aufrufen (Foto hochladen), bevor die personalisierte Vorschau abgerufen werden kann. Siehe [[Foto-Personalisierung]].

## Allgemein: Server neu starten nach `.env`-Änderungen

`.env` wird nur beim Start gelesen — nach jeder Änderung an Zugangsdaten den Server (`npm run dev`) neu starten.
