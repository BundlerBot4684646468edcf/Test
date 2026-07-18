'use client';

import { useEffect, useState } from 'react';
import api from '@/lib/api';
import {
  LineChart,
  Line,
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
  ownerPhotoUrl?: string;
  dailyBatchLimit: number;
}

interface ReviewEvent {
  id: string;
  newReviewCount: number;
  avgRating: number;
  totalReviews: number;
  checkedAt: string;
}

export default function DashboardPage() {
  const [business, setBusiness] = useState<Business | null>(null);
  const [reviewEvents, setReviewEvents] = useState<ReviewEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const businessId = localStorage.getItem('businessId');
    if (!businessId) return;

    loadBusiness(businessId);
  }, []);

  const loadBusiness = async (businessId: string) => {
    try {
      const response = await api.get(`/businesses/${businessId}`);
      setBusiness(response.data);
      if (response.data.reviewEvents) {
        setReviewEvents(response.data.reviewEvents);
      }
    } catch (err) {
      setError('Failed to load business');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-8">Loading...</div>;
  }

  if (!business) {
    return <div className="text-center text-red-600">{error}</div>;
  }

  const chartData = reviewEvents
    .slice()
    .reverse()
    .map((event) => ({
      date: new Date(event.checkedAt).toLocaleDateString('de-DE'),
      rating: event.avgRating,
      reviews: event.totalReviews,
    }));

  const latestEvent = reviewEvents[0];
  const previousEvent = reviewEvents[1];
  const reviewTrend = latestEvent && previousEvent
    ? latestEvent.totalReviews - previousEvent.totalReviews
    : 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-gray-900">{business.name}</h2>
        <p className="text-gray-600 mt-1">Owner: {business.ownerName}</p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm font-medium">Total Reviews</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {latestEvent?.totalReviews || 0}
          </p>
          {reviewTrend > 0 && (
            <p className="text-green-600 text-sm mt-2">
              +{reviewTrend} new this week
            </p>
          )}
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm font-medium">Average Rating</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {latestEvent?.avgRating?.toFixed(1) || '-'}
          </p>
          <p className="text-yellow-600 text-sm mt-2">★★★★★</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm font-medium">Daily Batch Limit</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {business.dailyBatchLimit}
          </p>
          <p className="text-gray-600 text-sm mt-2">messages per day</p>
        </div>
      </div>

      {/* Chart */}
      {chartData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Weekly Trend
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="reviews"
                stroke="#3b82f6"
                name="Total Reviews"
              />
              <Line
                type="monotone"
                dataKey="rating"
                stroke="#10b981"
                name="Avg Rating"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Setup Card */}
      {!business.googleReviewLink && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="font-semibold text-blue-900">Setup Required</h3>
          <p className="text-blue-700 text-sm mt-2">
            You need to set up your Google Place to start requesting reviews.
            Go to /api/businesses/{business.id}/find-place with your business
            name and address.
          </p>
        </div>
      )}
    </div>
  );
}
