---
tags: [mundpost, testing, feature]
---

# Foto-Personalisierung ("Name aufs Schild")

Gehört zu: [[Mundpost]] · [[Test-Checkliste]]

Das virale TikTok-Feature: Der Inhaber hält ein leeres Schild in die Kamera, Mundpost schreibt für jeden Kunden automatisch dessen Vornamen handschriftlich drauf, bevor die SMS/E-Mail rausgeht.

## Wie es technisch funktioniert

- Code: `src/services/photoPersonalization.ts`
- Rendering-Engine: `@napi-rs/canvas` (Canvas-API in Node, ohne Browser)
- Handschrift-Font: `@fontsource/caveat` (gebündelt, kein Systemfont nötig)
- UI-Text-Font: `@fontsource/inter` (gebündelt, seit Fix vom 18.07.2026 — vorher zeigte generisches `sans-serif` Kästchen/Tofu-Boxen auf manchen Windows-Systemen)
- Schriftgröße passt sich automatisch an: langer Name → kleinere Schrift, bis er aufs Schild passt (`personalizePhoto()`)

## Schild-Position (`SignBox`)

Alle Koordinaten sind **relativ** (0.0–1.0), damit sie bei jeder Bildauflösung passen:

```ts
{
  x: 0.5,        // Mittelpunkt horizontal (0=links, 1=rechts)
  y: 0.62,       // Mittelpunkt vertikal (0=oben, 1=unten)
  width: 0.55,   // Breite als Anteil der Bildbreite
  height: 0.22,  // Höhe als Anteil der Bildhöhe
  rotation: -3,  // Neigung in Grad (leichte Schräglage wie beim Original-Video)
}
```

Default-Werte, falls für einen Betrieb noch nichts eingestellt wurde: siehe `DEFAULT_SIGN_BOX`.

## Schritt für Schritt: echtes Foto einrichten

**1. Foto hochladen**
```
POST http://localhost:3000/api/businesses/:businessId/photo
Content-Type: multipart/form-data
Feld: photo = <Bilddatei, JPEG/PNG/WebP, max 5MB>
```

**2. Schild-Position einstellen**

Am einfachsten: Foto in einem Bildbearbeitungsprogramm (oder sogar Paint) öffnen, ungefähre Pixel-Koordinaten der 4 Schild-Ecken ablesen, dann in relative Werte umrechnen (`x_relativ = x_pixel / bildbreite`).

```
PUT http://localhost:3000/api/businesses/:businessId/photo/sign-box
Content-Type: application/json

{ "x": 0.5, "y": 0.6, "width": 0.5, "height": 0.2, "rotation": 0 }
```

**3. Vorschau ansehen**
```
GET http://localhost:3000/api/businesses/:businessId/photo/personalized?name=Anna
```

Sitzt der Name nicht richtig auf dem Schild? → Schritt 2 wiederholen mit angepassten Werten, bis es passt. Trial-and-Error ist hier normal — es gibt (noch) kein visuelles Einstell-Tool im Dashboard dafür.

## Wo es im echten Versand verwendet wird

`src/services/reviewQueue.ts` → Funktion `photoUrlFor()`:
- wird bei **jedem** Kunden aus der Warteschlange aufgerufen
- lädt das Original-Foto, rendert den Vornamen drauf, speichert das Ergebnis unter `personalized/<businessId>/<name>.png`
- bei SMS: als `mediaUrl` mitgeschickt (nur wenn öffentlich erreichbar, siehe [[Cloudflare R2]])
- bei E-Mail: inline im HTML eingebettet
- **Fallback:** falls irgendwas schiefgeht (Foto fehlt, Rendering-Fehler), wird einfach das unveränderte Originalfoto verwendet — der Versand bricht nicht ab

## Bekannte Grenzen

- Nur der **Vorname** wird geschrieben, kein Nachname (Datenschutz + Platz auf dem Schild)
- Sonderzeichen/Umlaute im Namen sollten mit Caveat funktionieren (lateinischer Zeichensatz), aber nicht mit z.B. kyrillischen Namen getestet
- Rotation nur starr einstellbar pro Betrieb (nicht pro Foto), falls mehrere Fotos mit unterschiedlicher Perspektive verwendet werden, müsste man das Datenmodell erweitern (aktuell: 1 Foto + 1 Schild-Position pro Betrieb)
