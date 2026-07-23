'use client';

import { useEffect, useState } from 'react';
import api from '@/lib/api';

interface Customer {
  id: string;
  firstName: string;
  phone?: string;
  email?: string;
  servedAt: string;
  source: string;
  optOut: number | boolean;
}

const CHANNELS: Array<{ key: 'email' | 'sms' | 'whatsapp'; label: string }> = [
  { key: 'email', label: 'E-Mail' },
  { key: 'sms', label: 'SMS' },
  { key: 'whatsapp', label: 'WhatsApp' },
];

const card: React.CSSProperties = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 14,
};

export default function CustomersPage() {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [consentGiven, setConsentGiven] = useState(false);
  const [channelFor, setChannelFor] = useState<Record<string, 'email' | 'sms' | 'whatsapp'>>({});
  const [sendingFor, setSendingFor] = useState<Record<string, boolean>>({});
  const [sentFor, setSentFor] = useState<Record<string, boolean>>({});

  const load = () => {
    const id = localStorage.getItem('businessId');
    if (!id) return;
    api
      .get(`/businesses/${id}/customers`)
      .then((r) => setCustomers(r.data.customers))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const upload = async () => {
    if (!file || !consentGiven) {
      setMsg({ ok: false, text: 'Bitte bestätige, dass die Kunden der Datenverarbeitung zugestimmt haben.' });
      return;
    }
    setUploading(true);
    setMsg(null);
    try {
      const id = localStorage.getItem('businessId');
      const fd = new FormData();
      fd.append('file', file);
      fd.append('source', 'past');
      fd.append('consent', 'true');
      const r = await api.post(`/businesses/${id}/customers/import`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMsg({ ok: true, text: `${r.data.imported} Kunden importiert${r.data.errors.length ? `, ${r.data.errors.length} Fehler` : ''}.` });
      setFile(null);
      setConsentGiven(false);
      load();
    } catch {
      setMsg({ ok: false, text: 'Upload fehlgeschlagen.' });
    } finally {
      setUploading(false);
    }
  };

  const sendRequest = async (customer: Customer) => {
    const channel = channelFor[customer.id] || (customer.email ? 'email' : 'sms');
    setSendingFor((s) => ({ ...s, [customer.id]: true }));
    setMsg(null);
    try {
      const id = localStorage.getItem('businessId');
      await api.post(`/businesses/${id}/review-requests`, {
        customerId: customer.id,
        channel,
      });
      setSentFor((s) => ({ ...s, [customer.id]: true }));
      setMsg({ ok: true, text: `Anfrage für ${customer.firstName} eingeplant (${channel}). Der Versand läuft alle 15 Minuten automatisch.` });
    } catch (err: any) {
      setMsg({
        ok: false,
        text: err?.response?.data?.error || `Anfrage für ${customer.firstName} fehlgeschlagen.`,
      });
    } finally {
      setSendingFor((s) => ({ ...s, [customer.id]: false }));
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
        Kunden
      </h1>

      <div style={card} className="p-6">
        <h2 className="text-sm font-semibold" style={{ color: 'var(--ink)' }}>CSV importieren</h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>
          Spalten: <code className="tnum">firstName, phone, email, servedAt, servedBy</code> (servedBy optional — Mitarbeiter:in für die persönliche Ansprache)
        </p>

        <div
          className="mt-4 rounded-xl p-6 text-center"
          style={{ border: '1.5px dashed var(--line)', background: 'var(--plane)' }}
        >
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm"
            style={{ color: 'var(--ink-2)' }}
          />
        </div>

        {file && (
          <div className="mt-4 space-y-3">
            <label className="flex items-start gap-3 p-3 rounded-lg" style={{ background: 'var(--plane)', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={consentGiven}
                onChange={(e) => setConsentGiven(e.target.checked)}
                className="mt-1"
              />
              <span style={{ color: 'var(--ink-2)', fontSize: '0.9rem', lineHeight: 1.5 }}>
                ✓ Ich bestätige, dass diese Kunden der Datenverarbeitung und dem Erhalt von Bewertungsanfragen zugestimmt haben
                (GDPR Datenschutz).
                <a href="/dashboard/privacy" className="ml-1" style={{ color: 'var(--brand)' }}>
                  Datenschutzerklärung lesen
                </a>
              </span>
            </label>
            <button
              onClick={upload}
              disabled={uploading || !consentGiven}
              className="w-full py-2.5 rounded-lg font-medium text-white transition-opacity disabled:opacity-60"
              style={{ background: 'var(--brand)' }}
            >
              {uploading ? 'Importiere…' : `„${file.name}" importieren`}
            </button>
          </div>
        )}

        {msg && (
          <div
            className="mt-4 p-3 rounded-lg text-sm"
            style={{
              color: msg.ok ? 'var(--good-ink)' : 'var(--critical)',
              background: msg.ok ? 'rgba(12,163,12,0.10)' : 'rgba(208,59,59,0.10)',
            }}
          >
            {msg.text}
          </div>
        )}
      </div>

      <div style={card} className="overflow-hidden">
        <div className="px-5 py-3.5" style={{ borderBottom: '1px solid var(--border)' }}>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--ink)' }}>
            {customers.length} {customers.length === 1 ? 'Kunde' : 'Kunden'}
          </h2>
        </div>

        {loading ? (
          <div className="p-8 text-center" style={{ color: 'var(--muted)' }}>Lädt…</div>
        ) : customers.length === 0 ? (
          <div className="p-8 text-center" style={{ color: 'var(--ink-2)' }}>
            Noch keine Kunden — lade eine CSV hoch.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Name', 'Telefon', 'E-Mail', 'Besuch', 'Status', 'Anfrage'].map((h) => (
                    <th key={h} className="text-left px-5 py-3 font-medium" style={{ color: 'var(--muted)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {customers.map((c) => {
                  const opted = c.optOut === 1 || c.optOut === true;
                  const channel = channelFor[c.id] || (c.email ? 'email' : 'sms');
                  const sending = !!sendingFor[c.id];
                  const sent = !!sentFor[c.id];
                  return (
                    <tr key={c.id} style={{ borderBottom: '1px solid var(--line)' }}>
                      <td className="px-5 py-3.5 font-medium" style={{ color: 'var(--ink)' }}>{c.firstName}</td>
                      <td className="px-5 py-3.5 tnum" style={{ color: 'var(--ink-2)' }}>{c.phone || '–'}</td>
                      <td className="px-5 py-3.5" style={{ color: 'var(--ink-2)' }}>{c.email || '–'}</td>
                      <td className="px-5 py-3.5 tnum" style={{ color: 'var(--ink-2)' }}>
                        {new Date(c.servedAt).toLocaleDateString('de-DE')}
                      </td>
                      <td className="px-5 py-3.5">
                        <span
                          className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium"
                          style={{
                            color: opted ? 'var(--critical)' : 'var(--good-ink)',
                            background: opted ? 'rgba(208,59,59,0.12)' : 'rgba(12,163,12,0.13)',
                          }}
                        >
                          {opted ? 'Abgemeldet' : 'Aktiv'}
                        </span>
                      </td>
                      <td className="px-5 py-3.5">
                        {opted ? (
                          <span style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>–</span>
                        ) : sent ? (
                          <span style={{ color: 'var(--good-ink)', fontSize: '0.85rem' }}>✓ Eingeplant</span>
                        ) : (
                          <div className="flex items-center gap-2">
                            <select
                              value={channel}
                              onChange={(e) =>
                                setChannelFor((s) => ({ ...s, [c.id]: e.target.value as any }))
                              }
                              className="px-2 py-1.5 rounded-md text-xs"
                              style={{ border: '1px solid var(--border)', background: 'var(--plane)', color: 'var(--ink)' }}
                            >
                              {CHANNELS.map((ch) => (
                                <option key={ch.key} value={ch.key}>
                                  {ch.label}
                                </option>
                              ))}
                            </select>
                            <button
                              onClick={() => sendRequest(c)}
                              disabled={sending}
                              className="px-3 py-1.5 rounded-md text-xs font-medium text-white disabled:opacity-60"
                              style={{ background: 'var(--brand)' }}
                            >
                              {sending ? 'Sendet…' : 'Anfrage senden'}
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
