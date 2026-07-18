'use client';

import { useEffect, useState } from 'react';
import api from '@/lib/api';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface Business {
  id: string;
  name: string;
  ownerName: string;
  googleReviewLink?: string;
  dailyBatchLimit: number;
}
interface ReviewEvent {
  newReviewCount: number;
  avgRating: number | null;
  totalReviews: number;
  checkedAt: string;
}

const card: React.CSSProperties = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 14,
};

function StatTile({
  label,
  value,
  sub,
  subColor,
}: {
  label: string;
  value: string;
  sub?: string;
  subColor?: string;
}) {
  return (
    <div style={card} className="p-5">
      <p className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--muted)' }}>
        {label}
      </p>
      <p className="mt-2 text-3xl font-semibold tnum" style={{ color: 'var(--ink)' }}>
        {value}
      </p>
      {sub && (
        <p className="mt-1.5 text-sm" style={{ color: subColor || 'var(--ink-2)' }}>
          {sub}
        </p>
      )}
    </div>
  );
}

export default function DashboardPage() {
  const [business, setBusiness] = useState<Business | null>(null);
  const [events, setEvents] = useState<ReviewEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const id = localStorage.getItem('businessId');
    if (!id) return;
    api
      .get(`/businesses/${id}`)
      .then((r) => {
        setBusiness(r.data);
        setEvents(r.data.reviewEvents || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return <div className="text-center py-16" style={{ color: 'var(--muted)' }}>Lädt…</div>;
  if (!business)
    return <div className="text-center py-16" style={{ color: 'var(--critical)' }}>Betrieb nicht gefunden</div>;

  const chartData = events
    .slice()
    .reverse()
    .map((e) => ({
      date: new Date(e.checkedAt).toLocaleDateString('de-DE', { day: '2-digit', month: '2-digit' }),
      reviews: e.totalReviews,
    }));

  const latest = events[0];
  const prev = events[1];
  const trend = latest && prev ? latest.totalReviews - prev.totalReviews : 0;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
          {business.name}
        </h1>
        <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>
          Inhaber:in {business.ownerName}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatTile
          label="Bewertungen gesamt"
          value={latest ? String(latest.totalReviews) : '0'}
          sub={trend > 0 ? `+${trend} diese Woche` : 'noch keine neuen'}
          subColor={trend > 0 ? 'var(--good-ink)' : 'var(--muted)'}
        />
        <StatTile
          label="Durchschnitt"
          value={latest?.avgRating ? latest.avgRating.toFixed(1) : '–'}
          sub="★ von 5"
          subColor="var(--warning)"
        />
        <StatTile
          label="Tageslimit"
          value={String(business.dailyBatchLimit)}
          sub="Nachrichten / Tag"
        />
      </div>

      {chartData.length > 0 ? (
        <div style={card} className="p-6">
          <h2 className="text-sm font-semibold mb-4" style={{ color: 'var(--ink)' }}>
            Bewertungen im Zeitverlauf
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
              <defs>
                <linearGradient id="revFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--brand)" stopOpacity={0.18} />
                  <stop offset="100%" stopColor="var(--brand)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} stroke="var(--line)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={{ stroke: 'var(--line)' }} tickLine={false} />
              <YAxis tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={false} tickLine={false} width={40} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  fontSize: 13,
                }}
                labelStyle={{ color: 'var(--ink-2)' }}
              />
              <Area
                type="monotone"
                dataKey="reviews"
                name="Bewertungen"
                stroke="var(--brand)"
                strokeWidth={2}
                fill="url(#revFill)"
                dot={false}
                activeDot={{ r: 5, fill: 'var(--brand)' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div style={card} className="p-8 text-center">
          <p className="text-sm" style={{ color: 'var(--ink-2)' }}>
            Noch keine Bewertungs-Daten. Sobald dein Google-Betrieb verknüpft ist, erscheint hier der Verlauf.
          </p>
        </div>
      )}

      {!business.googleReviewLink && (
        <div
          className="p-5 rounded-xl flex gap-3"
          style={{ background: 'var(--brand-50)', border: '1px solid color-mix(in srgb, var(--brand) 25%, transparent)' }}
        >
          <span aria-hidden style={{ color: 'var(--brand-600)' }}>ℹ️</span>
          <div>
            <p className="font-semibold text-sm" style={{ color: 'var(--brand-600)' }}>
              Google-Betrieb noch nicht verknüpft
            </p>
            <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>
              Verknüpfe deinen Betrieb mit Google, um den Bewertungs-Link zu erzeugen und den Versand zu starten.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
