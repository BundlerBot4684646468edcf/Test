'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';

const nav = [
  { href: '/dashboard', label: 'Übersicht' },
  { href: '/dashboard/customers', label: 'Kunden' },
  { href: '/dashboard/reviews', label: 'Bewertungen' },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [businessId, setBusinessId] = useState('');

  useEffect(() => {
    const id = localStorage.getItem('businessId');
    if (!id) router.push('/');
    else setBusinessId(id);
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem('businessId');
    router.push('/');
  };

  return (
    <div className="min-h-screen" style={{ background: 'var(--plane)' }}>
      <nav
        className="sticky top-0 z-10 backdrop-blur"
        style={{
          background: 'color-mix(in srgb, var(--surface) 85%, transparent)',
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <span
                className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white text-sm font-bold"
                style={{ background: 'var(--brand)' }}
              >
                M
              </span>
              <span className="font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
                Mundpost
              </span>
            </div>
            <div className="hidden md:flex items-center gap-1">
              {nav.map((item) => {
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                      color: active ? 'var(--brand-600)' : 'var(--ink-2)',
                      background: active ? 'var(--brand-50)' : 'transparent',
                    }}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <code
              className="hidden sm:block text-xs px-2.5 py-1 rounded-md tnum"
              style={{ background: 'var(--plane)', color: 'var(--muted)', border: '1px solid var(--border)' }}
            >
              {businessId.slice(0, 8)}…
            </code>
            <button
              onClick={handleLogout}
              className="text-sm font-medium px-3 py-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--ink-2)' }}
            >
              Abmelden
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-10">{children}</main>
    </div>
  );
}
