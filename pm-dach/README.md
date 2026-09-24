# PM Dach – Patrick Mittermair · Website-Konzept

Eine einzige Datei: `index.html` (kein Build nötig, einfach im Browser öffnen).

## Leitidee: „Egal, was von oben kommt.“

Die Website zeigt, was ein Dach leistet, statt es nur zu behaupten.

1. **Wetter-Hero (Erlebnis):** Ein live gezeichnetes Haus. Der Besucher wählt Regen, Schnee, Sturm oder Sonne. Das Dach hält alles ab (Schnee bleibt liegen, Rauch zieht aus dem Kamin, Blitze im Sturm) und zählt mit: „Vom Dach abgehalten: 1.284“. Der Mauszeiger steuert den Wind.
2. **Dachaufbau Schicht für Schicht (Information):** Beim Scrollen baut sich ein Steildach zusammen: Tragwerk → Dämmung → Unterdach → Lattung → Eindeckung → Spengler. Am Ende regnet es darauf. Kunden verstehen, wofür sie bezahlen.
3. **Dach-Anfrage in 60 Sekunden (Anfragen):** 4 Schritte mit großen Kacheln (Anliegen, Dachform, Zeitraum, Kontakt).
4. **Sprachassistent (wie elotec / bauprofi):** Im Browser (Web Speech API, kein Server nötig). Er fragt Anliegen, Dachform, Ort, Zeitraum, Name und Kontakt ab, fasst zusammen und gibt die Anfrage erst nach Bestätigung weiter. Er beantwortet Preis-, Standort-, Telefon- und „Wer steckt dahinter“-Fragen **nur aus den eingetragenen Firmendaten**. Ohne Mikrofon funktioniert er per Buttons oder Texteingabe.
5. **Zweisprachig DE / IT** (Südtirol), mobile Leiste mit Anrufen / Anfrage / Assistent.

## Was noch fehlt (bitte nur bestätigte Angaben)

Oben im `<script>` im Objekt `COMPANY`:

- `street`, `zip`, `phone`, `email`, `vat`: aus dem LVH-Eintrag übernehmen
- `email` ist der Empfänger der Anfragen

Außerdem:

- `SERVICES`: Leistungen mit PM Dach abgleichen, danach `confirmed: true`
- Porträtfoto Patrick Mittermair, Geschichte des Betriebs (Abschnitt „Wer hinter PM Dach steht“)
- **Echte Referenzfotos von PM Dach.** Die Bilder von gasserpaul.it gehören einem anderen Betrieb (Gasser Paul GmbH, St. Lorenzen) und dürfen hier nicht verwendet werden.
- Datenschutz- und Impressum-Seiten
- Live-Betrieb: in `submitLead()` die Anfrage an ein Formular-Backend / CRM senden (Stelle ist markiert)

Wenn alles eingetragen ist: `const DRAFT = false;` setzen, dann verschwinden alle gelben Markierungen.
