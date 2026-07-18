import express from 'express';
import twilio from 'twilio';
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

// Normalise a number the browser may have mangled (dropped '+', added spaces).
function normalizePhone(raw: string): string {
  let n = (raw || '').trim().replace(/[\s]/g, '');
  if (n.startsWith(' ')) n = n.trim();
  if (!n.startsWith('+')) n = '+' + n.replace(/^\+*/, '');
  return n;
}

async function doTestSms(to: string, res: express.Response) {
  const phone = normalizePhone(to);
  if (phone.length < 8) {
    return res.status(400).json({
      error:
        'Nummer fehlt. Öffne z.B. http://localhost:3000/api/setup/test-sms/393273042753',
    });
  }
  const result = await sendSMS({
    toPhone: phone,
    message:
      "Hallo! Das ist deine Mundpost-Test-SMS 🎉 Wenn du das liest, funktioniert der SMS-Versand wirklich. — Mundpost",
  });
  if (result.success) {
    res.json({ success: true, sentTo: phone, messageId: result.messageId });
  } else {
    res.status(500).json({ success: false, sentTo: phone, error: result.error });
  }
}

// Easiest: number in the path, no '+' or '?' needed.
// e.g. http://localhost:3000/api/setup/test-sms/393273042753
router.get('/test-sms/:number', async (req, res) => {
  await doTestSms(req.params.number, res);
});

// Also works with a query param: /api/setup/test-sms?to=+39...
router.get('/test-sms', async (req, res) => {
  await doTestSms((req.query.to as string) || '', res);
});

// GET /api/setup/sms-status/:sid — Ask Twilio the REAL delivery status of a
// message (Twilio "success" only means accepted, not delivered).
router.get('/sms-status/:sid', async (req, res) => {
  const sid = process.env.TWILIO_ACCOUNT_SID || '';
  const token = process.env.TWILIO_AUTH_TOKEN || '';
  if (!sid || !token) {
    return res.status(400).json({ error: 'Twilio not configured' });
  }
  try {
    const client = twilio(sid, token);
    const m = await client.messages(req.params.sid).fetch();
    res.json({
      status: m.status, // queued | sent | delivered | undelivered | failed
      errorCode: m.errorCode,
      errorMessage: m.errorMessage,
      to: m.to,
      from: m.from,
      dateSent: m.dateSent,
      hint:
        m.errorCode === 21608
          ? 'Trial-Konto: Zielnummer muss in Twilio verifiziert sein.'
          : m.errorCode === 30008 || m.status === 'undelivered'
          ? 'Vom Netz nicht zugestellt (oft US-Nummer -> Italien). Italienische/lokale Absendernummer oder Alphanumeric Sender ID nötig.'
          : undefined,
    });
  } catch (error: any) {
    res.status(500).json({ error: error?.message || String(error) });
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
