import express from 'express';
import { isGooglePlacesConfigured } from '../services/googlePlaces';
import { isMessagingConfigured } from '../services/messaging';

const router = express.Router();

// GET /api/setup/status — Check which services are configured
router.get('/status', (req, res) => {
  const googlePlaces = isGooglePlacesConfigured();
  const messaging = isMessagingConfigured();

  res.json({
    services: {
      googlePlaces: {
        configured: googlePlaces,
        status: googlePlaces ? '✅ Ready' : '❌ Missing GOOGLE_PLACES_API_KEY',
      },
      sms: {
        configured: messaging.sms,
        status: messaging.sms
          ? '✅ Ready'
          : '❌ Missing Twilio credentials',
      },
      email: {
        configured: messaging.email,
        status: messaging.email ? '✅ Ready' : '❌ Missing RESEND_API_KEY',
      },
    },
    allConfigured: googlePlaces && messaging.sms && messaging.email,
  });
});

export default router;
