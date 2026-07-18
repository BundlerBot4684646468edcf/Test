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
    if (!file) return;
    setUploading(true);
    setMsg(null);
    try {
      const id = localStorage.getItem('businessId');
      const fd = new FormData();
      fd.append('file', file);
      fd.append('source', 'past');
      const r = await api.post(`/businesses/${id}/customers/import`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMsg({ ok: true, text: `${r.data.imported} Kunden importiert${r.data.errors.length ? `, ${r.data.errors.length} Fehler` : ''}.` });
      setFile(null);
      load();
    } catch {
      setMsg({ ok: false, text: 'Upload fehlgeschlagen.' });
    } finally {
      setUploading(false);
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
          Spalten: <code className="tnum">firstName, phone, email, servedAt</code>
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
          <button
            onClick={upload}
            disabled={uploading}
            className="mt-4 w-full py-2.5 rounded-lg font-medium text-white transition-opacity disabled:opacity-60"
            style={{ background: 'var(--brand)' }}
          >
            {uploading ? 'Importiere…' : `„${file.name}" importieren`}
          </button>
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
                  {['Name', 'Telefon', 'E-Mail', 'Besuch', 'Status'].map((h) => (
                    <th key={h} className="text-left px-5 py-3 font-medium" style={{ color: 'var(--muted)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {customers.map((c) => {
                  const opted = c.optOut === 1 || c.optOut === true;
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
