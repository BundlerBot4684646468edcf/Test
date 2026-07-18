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

// GET /api/setup/test-sms-check/:number — Send AND poll delivery status in one
// browser hit, so we get the definitive result without copying the SID around.
router.get('/test-sms-check/:number', async (req, res) => {
  const accountSid = process.env.TWILIO_ACCOUNT_SID || '';
  const token = process.env.TWILIO_AUTH_TOKEN || '';
  if (!accountSid || !token) {
    return res.status(400).json({ error: 'Twilio not configured' });
  }

  let phone = (req.params.number || '').trim().replace(/\s/g, '');
  if (!phone.startsWith('+')) phone = '+' + phone.replace(/^\+*/, '');

  try {
    const client = twilio(accountSid, token);
    const msg = await client.messages.create({
      from: process.env.TWILIO_PHONE_NUMBER || '',
      to: phone,
      body:
        "Hallo! Mundpost-Test 🎉 Wenn du das liest, funktioniert der SMS-Versand. — Mundpost",
    });

    // Poll up to ~18s for a final status
    let status = msg.status;
    let errorCode: number | null = msg.errorCode ?? null;
    let errorMessage: string | null = msg.errorMessage ?? null;
    for (let i = 0; i < 9; i++) {
      if (['delivered', 'undelivered', 'failed'].includes(status)) break;
      await new Promise((r) => setTimeout(r, 2000));
      const m = await client.messages(msg.sid).fetch();
      status = m.status;
      errorCode = m.errorCode ?? null;
      errorMessage = m.errorMessage ?? null;
    }

    let hint: string | undefined;
    if (errorCode === 21608)
      hint = 'Trial: Zielnummer nicht verifiziert.';
    else if (errorCode === 30008 || status === 'undelivered')
      hint =
        'Vom Netz nicht zugestellt — typisch bei US-Absendernummer nach Italien. Für Italien brauchst du eine italienische/EU-Nummer oder Alphanumeric Sender ID (und ein aufgeladenes, kein Trial-Konto).';
    else if (errorCode === 30007)
      hint = 'Vom Carrier als Spam gefiltert.';

    res.json({ sid: msg.sid, to: phone, status, errorCode, errorMessage, hint });
  } catch (error: any) {
    res.status(500).json({ error: error?.message || String(error), code: error?.code });
  }
});

// GET /api/setup/test-photo/:name — Instant browser preview of the
// personalized "name on the sign" photo (demo backdrop, no upload needed).
router.get('/test-photo/:name', async (req, res) => {
  try {
    const { demoPhoto } = await import('../services/photoPersonalization');
    const name = (req.params.name || 'Anna').slice(0, 40);
    const png = await demoPhoto(name);
    res.type('png').send(png);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: String(error) });
  }
});

// GET /api/setup/test-email-photo/:email — Send a full "photo + text" review
// email, so you can see the personalised message with an owner photo in your inbox.
router.get('/test-email-photo/:email', async (req, res) => {
  const to = req.params.email || '';
  if (!to.includes('@')) {
    return res.status(400).json({
      error: 'Bitte E-Mail angeben, z.B. /api/setup/test-email-photo/alex@example.com',
    });
  }

  // For the demo we use a public sample portrait; your real owner photo would
  // be shown the same way once uploaded.
  const samplePhoto =
    (req.query.photo as string) ||
    'https://images.unsplash.com/photo-1633332755192-727a05c4013d?w=200&h=200&fit=crop';

  const html = buildReviewRequestHTML(
    'Anna',
    'Deinem Betrieb',
    'https://search.google.com/local/writereview?placeid=DEMO',
    'Marco',
    samplePhoto
  );

  const result = await sendEmail({
    toEmail: to,
    subject: 'Eine kurze Frage von Marco 🙏',
    html,
    fromName: 'Marco',
  });

  if (result.success) {
    res.json({ success: true, sentTo: to, messageId: result.messageId, note: 'Prüfe dein Postfach (auch Spam).' });
  } else {
    res.status(500).json({ success: false, error: result.error });
  }
});

// GET /api/setup/test-mms/:number — Send a test MMS with a public sample image,
// to confirm your Twilio number can deliver picture messages at all.
router.get('/test-mms/:number', async (req, res) => {
  const accountSid = process.env.TWILIO_ACCOUNT_SID || '';
  const token = process.env.TWILIO_AUTH_TOKEN || '';
  if (!accountSid || !token) {
    return res.status(400).json({ error: 'Twilio not configured' });
  }

  let phone = (req.params.number || '').trim().replace(/\s/g, '');
  if (!phone.startsWith('+')) phone = '+' + phone.replace(/^\+*/, '');

  // A known publicly-reachable image (Twilio's own demo owl).
  const imageUrl = (req.query.img as string) || 'https://demo.twilio.com/owl.png';

  try {
    const client = twilio(accountSid, token);
    const msg = await client.messages.create({
      from: process.env.TWILIO_PHONE_NUMBER || '',
      to: phone,
      body: 'Mundpost Foto-Test 📸 — wenn du ein Bild siehst, funktioniert MMS.',
      mediaUrl: [imageUrl],
    });

    let status = msg.status;
    let errorCode: number | null = msg.errorCode ?? null;
    for (let i = 0; i < 9; i++) {
      if (['delivered', 'undelivered', 'failed'].includes(status)) break;
      await new Promise((r) => setTimeout(r, 2000));
      const m = await client.messages(msg.sid).fetch();
      status = m.status;
      errorCode = m.errorCode ?? null;
    }

    res.json({
      sid: msg.sid,
      to: phone,
      imageUrl,
      status,
      errorCode,
      hint:
        status === 'undelivered' || errorCode
          ? 'MMS oft eingeschränkt bei Trial-Konten und US-Nummer -> Italien. Für echtes Foto-MMS: aufgeladenes Konto + MMS-fähige Nummer.'
          : undefined,
    });
  } catch (error: any) {
    res.status(500).json({ error: error?.message || String(error), code: error?.code });
  }
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
