'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [businessId, setBusinessId] = useState('');

  useEffect(() => {
    const id = localStorage.getItem('businessId');
    if (!id) {
      router.push('/');
    } else {
      setBusinessId(id);
    }
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem('businessId');
    router.push('/');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-8">
            <h1 className="text-2xl font-bold text-gray-900">
              Mundpost
            </h1>
            <div className="hidden md:flex space-x-6">
              <Link
                href="/dashboard"
                className="text-gray-600 hover:text-gray-900 font-medium"
              >
                Dashboard
              </Link>
              <Link
                href="/dashboard/customers"
                className="text-gray-600 hover:text-gray-900 font-medium"
              >
                Kunden
              </Link>
              <Link
                href="/dashboard/reviews"
                className="text-gray-600 hover:text-gray-900 font-medium"
              >
                Bewertungen
              </Link>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <code className="text-sm bg-gray-100 px-3 py-1 rounded text-gray-600">
              {businessId.slice(0, 8)}...
            </code>
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-gray-600 hover:text-gray-900 font-medium"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  );
}
