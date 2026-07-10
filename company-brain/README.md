# amstudio — Company Brain (Obsidian Vault)

Das gemeinsame Firmen-Gehirn von **amstudio** (KI-Rezeptionist für Friseursalons), gebaut für [Obsidian](https://obsidian.md).

## Öffnen
1. Obsidian installieren (kostenlos).
2. **„Ordner als Vault öffnen"** → diesen Ordner `company-brain/` auswählen.
3. Kern-Plugin **Vorlagen/Templates** aktivieren (Einstellungen → Kern-Plugins) — vorkonfiguriert auf `99-Templates/` mit `YYYY-MM-DD`-Datum.
4. `00-Home/Home.md` öffnen und anpinnen.

## Struktur
```
00-Home/        Startseite / Dashboard
01-Company/     Mission, Strategie, OKRs
02-Team/        Personen & Onboarding
03-Product/     Produkt-Doku & Roadmap (KI-Rezeptionist)
04-Meetings/    Eine Notiz pro Meeting (JJJJ-MM-TT Thema)
05-Projects/    Eine Notiz pro Projekt (Owner, Ziel, Ende)
06-Customers/   Leichtes CRM — eine Notiz pro Salon
07-Knowledge/   Dauerhaftes Wissen (Famulor-Setup, Prozesse)
08-Decisions/   Entscheidungs-Protokolle — warum wir was gewählt haben
99-Templates/   Vorlagen für alles oben
```

## Regeln des Vaults
- **Eine Notiz pro Entität** — Person, Meeting, Salon, Projekt, Entscheidung.
- **Großzügig verlinken** — `[[Wiki-Links]]` machen aus Ordnern ein Gehirn (Graph: `Strg+G`).
- **Entscheidungen aufschreiben** — alles schwer Umkehrbare bekommt einen Decision Record.
- **Index-Notizen sind die Landkarte** — jeder Ordner hat eine `… Index`-Notiz; neue Notiz = neue Zeile dort.

## Sync im Team
Das Vault liegt in Git. Committen und pushen wie Code; `.obsidian/workspace.json` (Fensterlayout pro Gerät) ist git-ignoriert, damit es keine Konflikte gibt.
