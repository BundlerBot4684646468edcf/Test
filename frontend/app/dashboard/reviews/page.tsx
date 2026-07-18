'use client';

import { useEffect, useState } from 'react';
import api from '@/lib/api';

interface ReviewRequest {
  id: string;
  customer: {
    firstName: string;
    email?: string;
    phone?: string;
  };
  channel: string;
  status: string;
  sentAt?: string;
  remindedAt?: string;
  createdAt: string;
}

const statusColors: Record<string, string> = {
  queued: 'bg-yellow-100 text-yellow-700',
  sent: 'bg-blue-100 text-blue-700',
  reminded: 'bg-purple-100 text-purple-700',
  reviewed: 'bg-green-100 text-green-700',
  opted_out: 'bg-red-100 text-red-700',
};

export default function ReviewsPage() {
  const [requests, setRequests] = useState<ReviewRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('');

  useEffect(() => {
    loadRequests();
  }, [filter]);

  const loadRequests = async () => {
    try {
      const businessId = localStorage.getItem('businessId');
      if (!businessId) return;

      const params = filter ? { status: filter } : {};
      const response = await api.get(
        `/businesses/${businessId}/review-requests`,
        { params }
      );
      setRequests(response.data.requests);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const stats = {
    total: requests.length,
    sent: requests.filter((r) => r.status === 'sent').length,
    reviewed: requests.filter((r) => r.status === 'reviewed').length,
    optedOut: requests.filter((r) => r.status === 'opted_out').length,
  };

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-gray-900">Review Requests</h2>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Total Requests</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{stats.total}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Sent</p>
          <p className="text-2xl font-bold text-blue-600 mt-1">{stats.sent}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Reviewed</p>
          <p className="text-2xl font-bold text-green-600 mt-1">
            {stats.reviewed}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Opted Out</p>
          <p className="text-2xl font-bold text-red-600 mt-1">
            {stats.optedOut}
          </p>
        </div>
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        {['', 'queued', 'sent', 'reviewed', 'opted_out'].map((status) => (
          <button
            key={status || 'all'}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              filter === status
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 border border-gray-200 hover:border-gray-300'
            }`}
          >
            {status || 'All'}
          </button>
        ))}
      </div>

      {/* List */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        {loading ? (
          <div className="p-6 text-center">Loading...</div>
        ) : requests.length === 0 ? (
          <div className="p-6 text-center text-gray-600">
            No review requests
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold">
                    Customer
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">
                    Channel
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">
                    Sent
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold">
                    Reminded
                  </th>
                </tr>
              </thead>
              <tbody>
                {requests.map((request) => (
                  <tr
                    key={request.id}
                    className="border-b border-gray-200 hover:bg-gray-50"
                  >
                    <td className="px-6 py-4 text-sm">
                      <div className="font-medium text-gray-900">
                        {request.customer.firstName}
                      </div>
                      <div className="text-gray-600 text-xs">
                        {request.customer.email || request.customer.phone}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {request.channel.toUpperCase()}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          statusColors[request.status]
                        }`}
                      >
                        {request.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {request.sentAt
                        ? new Date(request.sentAt).toLocaleDateString(
                            'de-DE'
                          )
                        : '-'}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {request.remindedAt
                        ? new Date(request.remindedAt).toLocaleDateString(
                            'de-DE'
                          )
                        : '-'}
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
