'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [businessId, setBusinessId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const r = await api.get(`/businesses/${businessId.trim()}`);
      if (r.data) {
        localStorage.setItem('businessId', businessId.trim());
        router.push('/dashboard');
      }
    } catch {
      setError('Betrieb nicht gefunden. Prüfe die ID.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4" style={{ background: 'var(--plane)' }}>
      <div className="w-full max-w-sm">
        <div className="flex items-center gap-2.5 mb-6 justify-center">
          <span
            className="inline-flex h-9 w-9 items-center justify-center rounded-xl text-white font-bold"
            style={{ background: 'var(--brand)' }}
          >
            M
          </span>
          <span className="text-xl font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
            Mundpost
          </span>
        </div>

        <div
          className="p-7"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16 }}
        >
          <h1 className="text-lg font-semibold" style={{ color: 'var(--ink)' }}>Anmelden</h1>
          <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>
            Gib die ID deines Betriebs ein.
          </p>

          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--ink-2)' }}>
                Business-ID
              </label>
              <input
                type="text"
                value={businessId}
                onChange={(e) => setBusinessId(e.target.value)}
                placeholder="z. B. 011fad11-ca93-…"
                className="w-full px-3.5 py-2.5 rounded-lg text-sm outline-none tnum"
                style={{ border: '1px solid var(--line)', background: 'var(--plane)', color: 'var(--ink)' }}
                required
              />
            </div>

            {error && (
              <div
                className="px-3.5 py-2.5 rounded-lg text-sm"
                style={{ color: 'var(--critical)', background: 'rgba(208,59,59,0.10)' }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-lg font-medium text-white transition-opacity disabled:opacity-60"
              style={{ background: 'var(--brand)' }}
            >
              {loading ? 'Prüfe…' : 'Anmelden'}
            </button>
          </form>
        </div>

        <p className="mt-5 text-center text-xs" style={{ color: 'var(--muted)' }}>
          Noch kein Betrieb? Lege einen über die API an — siehe QUICKSTART.md.
        </p>
      </div>
    </div>
  );
}
