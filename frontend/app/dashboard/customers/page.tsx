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
  optOut: boolean;
}

export default function CustomersPage() {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    loadCustomers();
  }, []);

  const loadCustomers = async () => {
    try {
      const businessId = localStorage.getItem('businessId');
      if (!businessId) return;

      const response = await api.get(`/businesses/${businessId}/customers`);
      setCustomers(response.data.customers);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setCsvFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!csvFile) return;

    setUploading(true);
    try {
      const businessId = localStorage.getItem('businessId');
      if (!businessId) return;

      const formData = new FormData();
      formData.append('file', csvFile);
      formData.append('source', 'past');

      const response = await api.post(
        `/businesses/${businessId}/customers/import`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );

      setMessage(
        `✅ Imported ${response.data.imported} customers. Errors: ${response.data.errors.length}`
      );
      setCsvFile(null);
      await loadCustomers();
    } catch (error) {
      setMessage('❌ Upload failed');
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-gray-900">Kunden</h2>

      {/* CSV Upload */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">CSV importieren</h3>
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="w-full"
            />
          </div>

          {csvFile && (
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 transition disabled:bg-gray-400"
            >
              {uploading ? 'Uploading...' : 'Import Customers'}
            </button>
          )}

          {message && (
            <div className={`p-3 rounded text-sm ${
              message.startsWith('✅')
                ? 'bg-green-50 text-green-700'
                : 'bg-red-50 text-red-700'
            }`}>
              {message}
            </div>
          )}
        </div>

        <div className="mt-6 pt-6 border-t text-sm text-gray-600">
          <p className="font-medium mb-2">CSV Format (with headers):</p>
          <pre className="bg-gray-100 p-3 rounded overflow-auto">
{`firstName,phone,email,servedAt
Marco,+393891234567,marco@example.com,2024-01-15
Anna,,anna@example.com,2024-01-14`}
          </pre>
        </div>
      </div>

      {/* Customers List */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">
            Customers ({customers.length})
          </h3>
        </div>

        {loading ? (
          <div className="p-6 text-center">Loading...</div>
        ) : customers.length === 0 ? (
          <div className="p-6 text-center text-gray-600">
            No customers yet. Upload a CSV file.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">
                    Phone
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">
                    Email
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">
                    Served
                  </th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {customers.map((customer) => (
                  <tr
                    key={customer.id}
                    className="border-b border-gray-200 hover:bg-gray-50"
                  >
                    <td className="px-6 py-4 text-sm text-gray-900">
                      {customer.firstName}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {customer.phone || '-'}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {customer.email || '-'}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {new Date(customer.servedAt).toLocaleDateString('de-DE')}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          customer.optOut
                            ? 'bg-red-100 text-red-700'
                            : 'bg-green-100 text-green-700'
                        }`}
                      >
                        {customer.optOut ? 'Opted Out' : 'Active'}
                      </span>
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
