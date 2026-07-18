'use client';

import { useEffect, useState } from 'react';
import api from '@/lib/api';

interface ReviewRequest {
  id: string;
  customer: { firstName: string; email?: string; phone?: string };
  channel: string;
  status: string;
  sentAt?: string;
  remindedAt?: string;
  createdAt: string;
}

const STATUS: Record<string, { label: string; fg: string; bg: string }> = {
  queued: { label: 'In Warteschlange', fg: '#8a5a00', bg: 'rgba(250,178,25,0.15)' },
  sent: { label: 'Gesendet', fg: 'var(--brand-600)', bg: 'var(--brand-50)' },
  reminded: { label: 'Erinnert', fg: '#a24a25', bg: 'rgba(236,131,90,0.15)' },
  reviewed: { label: 'Bewertet', fg: 'var(--good-ink)', bg: 'rgba(12,163,12,0.13)' },
  opted_out: { label: 'Abgemeldet', fg: 'var(--critical)', bg: 'rgba(208,59,59,0.12)' },
};

const card: React.CSSProperties = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 14,
};

function Badge({ status }: { status: string }) {
  const s = STATUS[status] || { label: status, fg: 'var(--ink-2)', bg: 'var(--plane)' };
  return (
    <span
      className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium"
      style={{ color: s.fg, background: s.bg }}
    >
      {s.label}
    </span>
  );
}

export default function ReviewsPage() {
  const [requests, setRequests] = useState<ReviewRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    const id = localStorage.getItem('businessId');
    if (!id) return;
    setLoading(true);
    api
      .get(`/businesses/${id}/review-requests`, { params: filter ? { status: filter } : {} })
      .then((r) => setRequests(r.data.requests))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [filter]);

  const stat = (s: string) => requests.filter((r) => r.status === s).length;

  const filters = [
    { key: '', label: 'Alle' },
    { key: 'queued', label: 'Wartend' },
    { key: 'sent', label: 'Gesendet' },
    { key: 'reviewed', label: 'Bewertet' },
    { key: 'opted_out', label: 'Abgemeldet' },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
        Bewertungsanfragen
      </h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Gesamt', value: requests.length, color: 'var(--ink)' },
          { label: 'Gesendet', value: stat('sent'), color: 'var(--brand-600)' },
          { label: 'Bewertet', value: stat('reviewed'), color: 'var(--good-ink)' },
          { label: 'Abgemeldet', value: stat('opted_out'), color: 'var(--critical)' },
        ].map((s) => (
          <div key={s.label} style={card} className="p-4">
            <p className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--muted)' }}>
              {s.label}
            </p>
            <p className="mt-1.5 text-2xl font-semibold tnum" style={{ color: s.color }}>
              {s.value}
            </p>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        {filters.map((f) => {
          const active = filter === f.key;
          return (
            <button
              key={f.key || 'all'}
              onClick={() => setFilter(f.key)}
              className="px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors"
              style={{
                color: active ? '#fff' : 'var(--ink-2)',
                background: active ? 'var(--brand)' : 'var(--surface)',
                border: `1px solid ${active ? 'var(--brand)' : 'var(--border)'}`,
              }}
            >
              {f.label}
            </button>
          );
        })}
      </div>

      <div style={card} className="overflow-hidden">
        {loading ? (
          <div className="p-8 text-center" style={{ color: 'var(--muted)' }}>Lädt…</div>
        ) : requests.length === 0 ? (
          <div className="p-8 text-center" style={{ color: 'var(--ink-2)' }}>
            Keine Anfragen in dieser Ansicht.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Kunde', 'Kanal', 'Status', 'Gesendet', 'Erinnert'].map((h) => (
                    <th key={h} className="text-left px-5 py-3 font-medium" style={{ color: 'var(--muted)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {requests.map((r) => (
                  <tr key={r.id} style={{ borderBottom: '1px solid var(--line)' }}>
                    <td className="px-5 py-3.5">
                      <div className="font-medium" style={{ color: 'var(--ink)' }}>{r.customer.firstName}</div>
                      <div className="text-xs" style={{ color: 'var(--muted)' }}>
                        {r.customer.email || r.customer.phone}
                      </div>
                    </td>
                    <td className="px-5 py-3.5" style={{ color: 'var(--ink-2)' }}>{r.channel.toUpperCase()}</td>
                    <td className="px-5 py-3.5"><Badge status={r.status} /></td>
                    <td className="px-5 py-3.5 tnum" style={{ color: 'var(--ink-2)' }}>
                      {r.sentAt ? new Date(r.sentAt).toLocaleDateString('de-DE') : '–'}
                    </td>
                    <td className="px-5 py-3.5 tnum" style={{ color: 'var(--ink-2)' }}>
                      {r.remindedAt ? new Date(r.remindedAt).toLocaleDateString('de-DE') : '–'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
