import { Router } from 'express';
import { customers, businesses } from '../db';

const router = Router();

// Minimal, self-contained HTML page — public, no login, mobile-friendly.
function page(title: string, bodyInner: string): string {
  return `<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f9fafb; color: #1f2937; margin: 0; padding: 2rem 1rem; line-height: 1.6; }
    .card { max-width: 560px; margin: 2rem auto; background: #fff; border: 1px solid #e5e7eb;
      border-radius: 14px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    h1 { font-size: 1.5rem; margin: 0 0 1rem; }
    h2 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }
    p, li { color: #4b5563; font-size: 0.95rem; }
    .ok { color: #0ca30c; font-weight: 600; }
    a { color: #0066cc; }
    .muted { color: #9ca3af; font-size: 0.8rem; }
  </style>
</head>
<body><div class="card">${bodyInner}</div></body>
</html>`;
}

// GET /u/:customerId — one-tap opt-out (works with one-way SMS sender IDs).
router.get('/u/:customerId', (req, res) => {
  try {
    const customer = customers.get(req.params.customerId);
    if (!customer) {
      return res
        .status(404)
        .send(page('Nicht gefunden', `<h1>Nicht gefunden</h1><p>Dieser Abmelde-Link ist ungültig oder abgelaufen.</p>`));
    }

    customers.optOut(customer.id);
    const business = businesses.get(customer.businessId);
    const name = business?.name ? ` von ${business.name}` : '';

    res.send(
      page(
        'Erfolgreich abgemeldet',
        `<h1>Du bist abgemeldet ✓</h1>
         <p class="ok">Hallo ${customer.firstName}, du erhältst keine weiteren Nachrichten${name} mehr.</p>
         <p>Deine Abmeldung wurde gespeichert. Falls du das aus Versehen gemacht hast, wende dich einfach direkt an den Betrieb.</p>
         <p class="muted">Mundpost · Datenschutz nach DSGVO</p>`
      )
    );
  } catch (error) {
    console.error('[UNSUBSCRIBE]', error);
    res.status(500).send(page('Fehler', `<h1>Fehler</h1><p>Bitte versuche es später erneut.</p>`));
  }
});

// GET /datenschutz — public privacy policy (linked from every message).
router.get('/datenschutz', (_req, res) => {
  res.send(
    page(
      'Datenschutzerklärung',
      `<h1>Datenschutzerklärung</h1>
       <p>Diese Nachrichten werden über <strong>Mundpost</strong> im Auftrag des jeweiligen Betriebs versendet.</p>

       <h2>1. Welche Daten</h2>
       <p>Vorname, Telefonnummer, E-Mail-Adresse, Besuchsdatum und ggf. der Name der bedienenden Person.</p>

       <h2>2. Zweck</h2>
       <p>Ausschließlich der Versand einer freundlichen Bitte um eine Google-Bewertung nach deinem Besuch.</p>

       <h2>3. Rechtsgrundlage</h2>
       <p>Deine Einwilligung, die du beim Besuch des Betriebs gegeben hast (Art. 6 Abs. 1 lit. a DSGVO).</p>

       <h2>4. Speicherdauer</h2>
       <p>Deine Daten werden spätestens 90 Tage nach der Anfrage gelöscht.</p>

       <h2>5. Deine Rechte</h2>
       <ul>
         <li>Jederzeit abmelden (Link in jeder Nachricht)</li>
         <li>Auskunft über deine gespeicherten Daten</li>
         <li>Löschung deiner Daten („Recht auf Vergessenwerden")</li>
       </ul>

       <h2>6. Datensicherheit</h2>
       <p>Übertragung verschlüsselt (HTTPS). SMS über Twilio, E-Mail über Resend, Hosting über Railway — alle mit Auftragsverarbeitungsvertrag (DSGVO-konform).</p>

       <h2>7. Kontakt</h2>
       <p>Bei Fragen wende dich direkt an den Betrieb, der dich bedient hat.</p>

       <p class="muted">Stand: Juli 2026</p>`
    )
  );
});

export default router;
