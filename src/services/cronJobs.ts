import cron from 'node-cron';
import { processReviewQueue } from './reviewQueue';
import { updateReviewMetrics } from './reviewMetrics';

export function startCronJobs() {
  // Process review queue every 15 minutes so sends land close to each
  // customer's conversion window (servedAt + sendDelayHours).
  cron.schedule('*/15 * * * *', () => {
    console.log('[CRON] Running review queue processor...');
    processReviewQueue().catch(console.error);
  });

  // Update review metrics daily at 2 AM
  cron.schedule('0 2 * * *', () => {
    console.log('[CRON] Running review metrics update...');
    updateReviewMetrics().catch(console.error);
  });

  console.log('✅ Cron jobs started');
}
