import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import businessesRouter from './routes/businesses';
import setupRouter from './routes/setup';
import { startCronJobs } from './services/cronJobs';

dotenv.config();

const app = express();

app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:3001' }));
app.use(express.json());

// Uploaded photos (owner photos) are served from local disk. Must match the
// directory fileStorage writes to (UPLOADS_DIR on a mounted volume in prod).
app.use('/uploads', express.static(process.env.UPLOADS_DIR || 'uploads'));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// API Routes
app.use('/api/businesses', businessesRouter);
app.use('/api/setup', setupRouter);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Mundpost server running on http://localhost:${PORT}`);
  startCronJobs();
});
