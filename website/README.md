# AM Studio — Website Redesign

A redesign concept for **amstudio.ink** — an AI agency building chatbot + phone
assistants for local businesses (South Tyrol / Alto Adige). Built as a
self-contained static site with **no build step and no external dependencies**,
so it runs anywhere by just opening `index.html`.

## What's inside

| File | Purpose |
|------|---------|
| `index.html` | Page markup (German primary, IT toggle) |
| `styles.css` | Premium dark theme, gradient mesh, glassmorphism, fully responsive |
| `script.js`  | Interactive logic — the live demo, language toggle, animations |

## Sections

1. **Hero** — headline, floating chat/SMS/call preview cards, key stats.
2. **Live demo (centerpiece)** — a working chatbot next to a phone mockup.
   Type or tap **`appuntamento`** and watch the full flow play out:
   - the assistant books a **19:30 table** →
   - a **calendar event** appears on the phone →
   - an **SMS confirmation** + a **review request** fire (booked + review together) →
   - an **incoming call** from the *Gerola* number (accept / decline).
   Reset with the button below the phone.
3. **Video** — styled poster + play button for the 90-second demo-call clip.
4. **Lösungen / Ablauf / Stats** — services, 3-step process, animated counters.
5. **Contact CTA + footer.**

## Run it

```bash
# just open the file
open website/index.html
# …or serve it
npx serve website
```

## Notes / assumptions

- The live site (`amstudio.ink`) is blocked by this environment's network
  policy, so the redesign was built from the brief rather than the existing
  page. Brand name, palette, and copy are placeholders that are easy to swap.
- Fonts load from Google Fonts when available and fall back to a system stack
  offline — no hard dependency.
- The demo phone number, business name (*Ristorante Gerola*) and review/SMS
  copy are illustrative; wire them to real data when integrating.
- All phone/calendar/SMS UI is rendered in HTML/CSS (no image assets), so there
  are no broken images and nothing to download.
