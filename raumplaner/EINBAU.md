# Raumplaner 3D – Einbau in `ralser-website`

Echter WebGL-Planer (three.js) als Ersatz für die gezeichnete Isometrie.
Getestet gegen Astro 7.3.4 und three 0.186.1 – beide stehen bereits in deiner
`package.json`.

## Dateien

| Datei | Zweck |
|---|---|
| `src/scripts/raumplaner-szene.ts` | Die 3D-Szene. Kein DOM, keine UI – nur `new Raumplaner(el)`, `.aktualisieren({…})`, `.start()`, `.bild()`, `.aufraeumen()`. |
| `src/components/Raumplaner.astro` | Bedienfeld, Styles und der Lade-Code. Bindet die Szene per `import()` ein. |

Beide 1:1 in dein Projekt kopieren, dann im bestehenden Abschnitt die alte
Zeichnung ersetzen:

```astro
---
import Raumplaner from '../components/Raumplaner.astro';
---
<Raumplaner vorschau="/bilder/raumplaner-vorschau.jpg" />
```

`three` ist aktuell eine `devDependency`. Da es in den Client-Bundle wandert,
gehört es zu den `dependencies`:

```
npm i three@^0.186.1 && npm i -D @types/three@^0.186.0
```

(Für `astro build` funktioniert beides, aber die Einordnung stimmt sonst nicht.)

## Was schon berücksichtigt ist

**CSP.** `astro build` erzeugt aus dem `<script>` der Komponente eine eigene
Datei, kein Inline-Script – `script-src 'self'` aus `public/_headers` bleibt
gültig. Geprüft: 0 Inline-Scripts im Build.

**Ladegewicht.** three.js landet in einem eigenen Chunk und wird erst geholt,
wenn der Abschnitt in Sichtweite kommt (`IntersectionObserver`, 200 px Vorlauf).

```
Raumplaner…script…js     1,5 kB gzip   (immer)
raumplaner-szene…js    139,0 kB gzip   (erst beim Scrollen)
```

Bis dahin steht das Vorschaubild. `assetsInlineLimit: 0` bleibt unangetastet.

**Ohne WebGL** (alte Browser, abgeschaltetes GL) bleibt das Vorschaubild stehen
und der Hinweis wechselt auf „Ansicht als Bild". Kein Absturz, keine leere Box.

**Speicher.** Jede Änderung baut die Möbel neu; `leeren()` gibt Geometrien,
Materialien und Texturen frei. 40 Raumwechsel hintereinander getestet – der
WebGL-Kontext überlebt.

**Mobil.** `touch-action: none` auf dem Canvas, sonst kämpft das Drehen gegen
das Seiten-Scrollen. Unter 62rem rutscht die Steuerung unter die Bühne.

## Anbindung ans Anfrageformular

Bei jeder Änderung feuert am Abschnitt ein Ereignis (steigt bis `document` auf):

```js
document.addEventListener('raumplaner:aenderung', (e) => {
  // e.detail = { raum, breite, tiefe, holz, front, glas, metall, licht }
});
```

Beim Absenden zusätzlich das Bild der aktuellen Ansicht mitschicken – das ist
erfahrungsgemäß das, was im Vertrieb tatsächlich gebraucht wird:

```js
const stand = window.raumplanerStand();
// { …Konfiguration, bild: "data:image/jpeg;base64,…" }  ≈ 69 kB
```

JPEG statt PNG, weil dieselbe Ansicht als PNG rund 470 kB wiegt. Dein
`/api/*`-Worker müsste das Feld noch annehmen und in die Mail hängen –
das habe ich nicht angefasst, weil ich `worker/index.ts` nicht kenne.

## Räume

Alle vier Varianten aus dem Bedienfeld sind gebaut, jeweils parametrisch aus
Quadern (Maße stufenlos, deshalb keine importierten Modelle):

- **Küche & Essen** – Unterschrankzeile, Hängeschränke, Hochschrank, Kochfeld,
  Insel (erscheint ab 3,0 m Raumtiefe)
- **Hotelzimmer** – Kopfteil, Bett, Nachttische mit Leseleisten, Schrank,
  Schreibtisch (nur wenn rechts wirklich Platz bleibt)
- **Wohnzimmer** – Regalwand mit beleuchteten Böden, Sofa, Couchtisch
- **Laden & Theke** – offenes Wandregal, L-förmige Theke mit Lichtkante

## Noch offen

- `public/bilder/raumplaner-vorschau.jpg` anlegen (Standbild der Küche). In
  `vorschau/ui-kueche.png` liegt ein passender Screenshot zum Zuschneiden.
- Die Möbelproportionen sind Normmaße (Arbeitshöhe 88 cm, Korpustiefe 62 cm).
  Wenn die Tischlerei anders baut, stehen die Werte gesammelt oben in den
  jeweiligen Raum-Methoden.
