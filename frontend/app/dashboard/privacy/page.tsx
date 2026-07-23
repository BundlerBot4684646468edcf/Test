'use client';

export default function PrivacyPage() {
  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--ink)' }}>
          Datenschutz & Rechtliches
        </h1>
        <p className="mt-2 text-lg" style={{ color: 'var(--ink-2)' }}>
          Informationen zum Umgang mit Kundendaten
        </p>
      </div>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold" style={{ color: 'var(--ink)' }}>
          Mundpost Datenschutzerklärung
        </h2>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            1. Welche Daten werden erhoben?
          </h3>
          <ul className="space-y-2" style={{ color: 'var(--ink-2)' }}>
            <li>• Kundenname (Vorname)</li>
            <li>• Telefonnummer</li>
            <li>• E-Mail-Adresse</li>
            <li>• Besuchsdatum des Salons</li>
            <li>• Name des Mitarbeiters (falls vorhanden)</li>
          </ul>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            2. Zweck der Datenverarbeitung
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6 }}>
            Die Daten werden ausschließlich verwendet, um automatisierte SMS- und E-Mail-Benachrichtigungen
            zu versenden, die Kunden bitten, eine Google-Bewertung zu verfassen. Die Nachrichten enthalten:
          </p>
          <ul className="space-y-2 mt-3" style={{ color: 'var(--ink-2)' }}>
            <li>• Personalisierte Grußbotschaft vom Salon-Inhaber</li>
            <li>• Foto des Inhabers (falls vorhanden)</li>
            <li>• Link zur Google-Bewertungsseite</li>
          </ul>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            3. Speicherdauer
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6 }}>
            Kundendaten werden <strong>90 Tage nach Absenden der Bewertungsanfrage</strong> automatisch gelöscht.
            Danach sind die Daten im System nicht mehr vorhanden. Reviews, die Kunden geschrieben haben,
            bleiben natürlich auf Google erhalten.
          </p>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            4. Rechtsgrundlage (GDPR)
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6 }}>
            Mundpost verarbeitet Kundendaten auf Basis der <strong>Einwilligung des Salon-Besitzers</strong>.
            Der Salon-Besitzer bestätigt beim CSV-Import, dass die Kunden der Datenverarbeitung zugestimmt haben.
          </p>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            5. Kundenrechte
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6 }}>
            Kunden können jederzeit:
          </p>
          <ul className="space-y-2 mt-3" style={{ color: 'var(--ink-2)' }}>
            <li>• <strong>Abmelden:</strong> Antworte "STOP" auf SMS oder klicke den Abmelde-Link in E-Mails</li>
            <li>• <strong>Daten anfragen:</strong> Kontaktiere den Salon-Besitzer</li>
            <li>• <strong>Daten löschen:</strong> Fordere sofortige Löschung an</li>
          </ul>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            6. Datensicherheit
          </h3>
          <ul className="space-y-2" style={{ color: 'var(--ink-2)' }}>
            <li>• Daten werden auf EU-Servern (Railway) gespeichert</li>
            <li>• Verschlüsselte Übertragung (HTTPS/TLS)</li>
            <li>• SMS über Twilio (GDPR-konform, DPA vorhanden)</li>
            <li>• E-Mail über Resend (GDPR-konform, DPA vorhanden)</li>
            <li>• Zugriff nur durch autorisiertes Salon-Personal</li>
          </ul>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            7. Drittparteien (Datenverarbeiter)
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6, marginBottom: '1rem' }}>
            Mundpost arbeitet mit folgenden GDPR-konformen Diensten:
          </p>
          <ul className="space-y-2" style={{ color: 'var(--ink-2)' }}>
            <li>• <strong>Twilio</strong> (SMS-Versand, USA, Privacy Shield + DPA)</li>
            <li>• <strong>Resend</strong> (E-Mail-Versand, USA, GDPR-konform)</li>
            <li>• <strong>Railway</strong> (Server-Hosting, EU, DSGVO-konform)</li>
          </ul>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            8. Kontakt für Datenschutzfragen
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6 }}>
            Fragen zu Datenschutz? Kontaktiere direkt deinen Salon-Besitzer oder den Mundpost-Support.
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold" style={{ color: 'var(--ink)' }}>
          Für Salon-Besitzer
        </h2>

        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.5rem', borderLeft: '4px solid var(--brand)' }}>
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--ink)' }}>
            Deine Verantwortung als Salon-Besitzer
          </h3>
          <p style={{ color: 'var(--ink-2)', lineHeight: 1.6, marginBottom: '1rem' }}>
            Als Verantwortlicher für die Datenverarbeitung in deinem Salon musst du:
          </p>
          <ul className="space-y-2" style={{ color: 'var(--ink-2)' }}>
            <li>✓ Sicherstellen, dass Kunden der Datenverarbeitung zugestimmt haben</li>
            <li>✓ Eine Datenschutzerklärung im Salon aushängen</li>
            <li>✓ Abmelderequests von Kunden respektieren</li>
            <li>✓ Kundenanfragen zu ihren Daten beantworten</li>
            <li>✓ Mundpost informieren, wenn ein Kunde sich abmelden möchte</li>
          </ul>
        </div>
      </section>

      <p className="text-sm" style={{ color: 'var(--muted)' }}>
        Letzte Aktualisierung: Juli 2026 | Version 1.0
      </p>
    </div>
  );
}
