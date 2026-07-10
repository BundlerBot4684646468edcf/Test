# Company Brain (Obsidian Vault)

A shared knowledge base for the company, built for [Obsidian](https://obsidian.md).

## Open it
1. Install Obsidian (free).
2. **Open folder as vault** → select this `company-brain/` directory.
3. Enable the core **Templates** plugin (Settings → Core plugins) — it's preconfigured to use `99-Templates/` with `YYYY-MM-DD` dates.
4. Open `00-Home/Home.md` and pin it.

## Structure
```
00-Home/        Entry-point dashboard
01-Company/     Mission, strategy, OKRs
02-Team/        People & onboarding
03-Product/     Product docs & roadmap
04-Meetings/    One note per meeting (YYYY-MM-DD Topic)
05-Projects/    One note per project (has an owner, goal, end date)
06-Customers/   Lightweight CRM — one note per hotel/account
07-Knowledge/   Evergreen how-tos and processes
08-Decisions/   Decision records — why we chose what we chose
99-Templates/   Templates for all of the above
```

## Rules of the vault
- **One note per entity** — person, meeting, customer, project, decision.
- **Link generously** — `[[wiki links]]` are what make it a brain, not a folder.
- **Write decisions down** — anything hard to reverse gets a Decision Record.
- **Index notes are the map** — each folder has an `… Index` note; add a row when you add a note.

## Syncing with the team
This vault lives in git. Commit and push like code; `.obsidian/workspace.json` (per-device layout) is git-ignored so you won't conflict on window state.
