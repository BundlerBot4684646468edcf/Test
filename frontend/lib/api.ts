import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add auth token if available
api.interceptors.request.use((config) => {
  const businessId = localStorage.getItem('businessId');
  if (businessId) {
    config.headers['X-Business-ID'] = businessId;
  }
  return config;
});

export default api;
