---
tags: [mundpost, setup, storage]
---

# Cloudflare R2 (Foto-Speicher für MMS)

Gehört zu: [[Mundpost]] · [[API Keys besorgen]] · [[Foto-Personalisierung]]

## Wozu

Twilio kann für MMS nur Bilder abrufen, die über eine **öffentlich erreichbare `https://`-URL** verfügbar sind. `localhost`-URLs (das lokale `/uploads`-Verzeichnis) funktionieren dafür nicht. Cloudflare R2 ist der S3-kompatible Objektspeicher, der diese öffentliche URL liefert.

**Ohne R2:** alles funktioniert außer Foto-**MMS** an echte Handynummern — E-Mail mit eingebettetem Foto, Dashboard-Vorschau und alle `/api/setup/test-photo`-Endpunkte laufen komplett lokal.

## Code

`src/services/fileStorage.ts`:
- `isStorageConfigured()` prüft, ob alle 5 R2-Variablen gesetzt sind
- `uploadFile()` schreibt lokal auf Platte, **wenn R2 nicht konfiguriert ist** → gibt `http://localhost:3000/uploads/...` zurück
- ist R2 konfiguriert, wird stattdessen zum Bucket hochgeladen → gibt die öffentliche R2-URL zurück
- **kein Code-Wechsel nötig** — nur `.env` befüllen, der Rest läuft automatisch um

## Einrichtung

1. https://dash.cloudflare.com/ → R2 aktivieren (im Free-Tier nutzbar)
2. Bucket erstellen
3. Bucket-Einstellungen → "Public Access" aktivieren → man bekommt eine `https://pub-xxxx.r2.dev`-URL (oder eigene Domain verbinden)
4. "Manage R2 API Tokens" → Access Key + Secret erzeugen
5. In `.env`:
   ```
   R2_ACCOUNT_ID=...
   R2_ACCESS_KEY_ID=...
   R2_SECRET_ACCESS_KEY=...
   R2_BUCKET_NAME=...
   R2_PUBLIC_BASE_URL=https://pub-xxxx.r2.dev
   ```
6. Server neu starten

## Test

Nach Einrichtung: Foto neu hochladen (`POST /photo`) → in der Antwort sollte jetzt eine `https://pub-....r2.dev/...`-URL stehen statt `http://localhost:3000/...`. Dann `GET /api/setup/test-mms/:number` probieren.

## Priorität

Nicht dringend — nur relevant, sobald echte SMS-Fotoversände (nicht nur E-Mail) in Produktion gehen sollen. Siehe [[Roadmap]].
