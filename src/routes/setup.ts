import express from 'express';
import { isGooglePlacesConfigured } from '../services/googlePlaces';
import {
  isMessagingConfigured,
  sendSMS,
  sendEmail,
  buildReviewRequestHTML,
} from '../services/messaging';

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
        status: messaging.sms ? '✅ Ready' : '❌ Missing Twilio credentials',
      },
      email: {
        configured: messaging.email,
        status: messaging.email ? '✅ Ready' : '❌ Missing RESEND_API_KEY',
      },
    },
    allConfigured: googlePlaces && messaging.sms && messaging.email,
  });
});

// GET /api/setup/test-sms?to=+39...  — Send a single real test SMS.
// Handy for a first end-to-end check straight from the browser.
router.get('/test-sms', async (req, res) => {
  const to = (req.query.to as string) || '';
  if (!to.startsWith('+')) {
    return res.status(400).json({
      error: 'Bitte ?to=+Ländervorwahl... angeben, z.B. /api/setup/test-sms?to=+393271234567',
    });
  }

  const result = await sendSMS({
    toPhone: to,
    message:
      "Hallo! Das ist deine Mundpost-Test-SMS 🎉 Wenn du das liest, funktioniert der SMS-Versand wirklich. — Mundpost",
  });

  if (result.success) {
    res.json({ success: true, sentTo: to, messageId: result.messageId });
  } else {
    res.status(500).json({ success: false, error: result.error });
  }
});

// GET /api/setup/test-email?to=name@example.com — Send a single real test email.
router.get('/test-email', async (req, res) => {
  const to = (req.query.to as string) || '';
  if (!to.includes('@')) {
    return res.status(400).json({
      error: 'Bitte ?to=deine@email.de angeben, z.B. /api/setup/test-email?to=alex@example.com',
    });
  }

  const html = buildReviewRequestHTML(
    'Test',
    'Deinem Betrieb',
    'https://search.google.com/local/writereview?placeid=DEMO',
    'Mundpost'
  );

  const result = await sendEmail({
    toEmail: to,
    subject: 'Mundpost Test-E-Mail 🎉',
    html,
    fromName: 'Mundpost',
  });

  if (result.success) {
    res.json({ success: true, sentTo: to, messageId: result.messageId });
  } else {
    res.status(500).json({ success: false, error: result.error });
  }
});

export default router;
