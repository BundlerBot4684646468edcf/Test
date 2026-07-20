import { Resend } from 'resend';
import twilio from 'twilio';

// Clients are created lazily so that credentials from .env are always read
// after the environment has finished loading, regardless of import order.
let twilioClient: ReturnType<typeof twilio> | null = null;
let twilioTriedFor = '';
let resendClient: Resend | null = null;
let resendTriedFor = '';

function getTwilio(): ReturnType<typeof twilio> | null {
  const sid = process.env.TWILIO_ACCOUNT_SID || '';
  const token = process.env.TWILIO_AUTH_TOKEN || '';
  const fingerprint = sid + ':' + token;
  if (fingerprint !== twilioTriedFor) {
    twilioTriedFor = fingerprint;
    twilioClient = sid && token ? twilio(sid, token) : null;
  }
  return twilioClient;
}

function getResend(): Resend | null {
  const key = process.env.RESEND_API_KEY || '';
  if (key !== resendTriedFor) {
    resendTriedFor = key;
    resendClient = key ? new Resend(key) : null;
  }
  return resendClient;
}

export interface SMSPayload {
  toPhone: string;
  message: string;
  mediaUrl?: string;
}

export interface EmailPayload {
  toEmail: string;
  subject: string;
  html: string;
  fromName?: string;
}

export async function sendSMS(payload: SMSPayload): Promise<{
  success: boolean;
  messageId?: string;
  error?: string;
}> {
  const client = getTwilio();
  if (!client) {
    console.warn(
      '[SMS] Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER'
    );
    return { success: false, error: 'Twilio not configured' };
  }

  try {
    const messageData: any = {
      body: payload.message,
      from: process.env.TWILIO_PHONE_NUMBER || '',
      to: payload.toPhone,
    };
    // Only attach media if it is a real public https URL Twilio can fetch.
    // A non-public placeholder (mock storage) would make Twilio reject the
    // whole message, so we fall back to a text-only SMS in that case.
    const isPublic =
      !!payload.mediaUrl &&
      payload.mediaUrl.startsWith('https://') &&
      !payload.mediaUrl.includes('mock-storage.local');
    if (isPublic) {
      messageData.mediaUrl = [payload.mediaUrl];
    } else if (payload.mediaUrl) {
      console.warn(
        '[SMS] Photo URL is not publicly reachable — sending text-only SMS. ' +
          'Configure R2 storage to enable photo-MMS.'
      );
    }

    try {
      const message = await client.messages.create(messageData);
      console.log(`✅ SMS sent: ${message.sid} to ${payload.toPhone}`);
      return { success: true, messageId: message.sid };
    } catch (error) {
      // MMS is not deliverable on many routes (e.g. US number -> Italy).
      // Rather than losing the message, retry as a plain text SMS.
      if (messageData.mediaUrl) {
        console.warn('[SMS] MMS rejected, retrying text-only:', error);
        delete messageData.mediaUrl;
        const message = await client.messages.create(messageData);
        console.log(`✅ SMS sent (text-only fallback): ${message.sid} to ${payload.toPhone}`);
        return { success: true, messageId: message.sid };
      }
      throw error;
    }
  } catch (error) {
    console.error('❌ SMS error:', error);
    return { success: false, error: String(error) };
  }
}

export async function sendEmail(payload: EmailPayload): Promise<{
  success: boolean;
  messageId?: string;
  error?: string;
}> {
  const client = getResend();
  if (!client) {
    console.warn('[EMAIL] Resend not configured. Set RESEND_API_KEY');
    return { success: false, error: 'Resend not configured' };
  }

  try {
    const result = await client.emails.send({
      from: `${payload.fromName || 'Mundpost'} <noreply@resend.dev>`,
      to: payload.toEmail,
      subject: payload.subject,
      html: payload.html,
    });

    if (result.error) {
      console.error('❌ Email error:', result.error);
      return { success: false, error: result.error.message };
    }

    console.log(`✅ Email sent: ${result.data?.id} to ${payload.toEmail}`);
    return { success: true, messageId: result.data?.id };
  } catch (error) {
    console.error('❌ Email error:', error);
    return { success: false, error: String(error) };
  }
}

// Template builders

// "Danke, dass du bei Lisa warst" beats a generic thank-you — but only when
// the staff member is someone other than the sender themselves.
function visitLine(servedBy: string | undefined | null, ownerName: string): string {
  return servedBy && servedBy.trim() && servedBy.trim() !== ownerName.trim()
    ? `Danke, dass du bei ${servedBy.trim()} warst! 🙏`
    : `Danke, dass du bei uns warst! 🙏`;
}

export function buildReviewRequestSMS(
  firstName: string,
  businessName: string,
  reviewLink: string,
  ownerName: string,
  servedBy?: string | null
): string {
  return `Hallo ${firstName},\n\nIch bin's, ${ownerName} von ${businessName}. ${visitLine(servedBy, ownerName)}\n\nHättest du vielleicht 30 Sekunden Zeit für eine kurze Google-Bewertung? Das bedeutet uns wirklich viel.\n\n${reviewLink}`;
}

export function buildReviewRequestHTML(
  firstName: string,
  businessName: string,
  reviewLink: string,
  ownerName: string,
  ownerPhotoUrl?: string,
  servedBy?: string | null
): string {
  // Large enough that the handwritten name on the sign is readable.
  const photoSection = ownerPhotoUrl
    ? `<img src="${ownerPhotoUrl}" alt="${ownerName}" style="width: 100%; max-width: 380px; border-radius: 12px; margin-bottom: 1rem;" />`
    : '';

  return `
    <html>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="max-width: 500px; margin: 0 auto; padding: 2rem;">
          ${photoSection}
          <h2>Hallo ${firstName}!</h2>
          <p>Ich bin's, ${ownerName} von ${businessName}. ${visitLine(servedBy, ownerName)}</p>
          <p>Hättest du vielleicht 30 Sekunden Zeit für eine kurze Google-Bewertung? Das bedeutet uns wirklich viel.</p>
          <div style="margin: 2rem 0;">
            <a href="${reviewLink}" style="background-color: #1f2937; color: white; padding: 0.75rem 1.5rem; border-radius: 0.375rem; text-decoration: none; display: inline-block;">
              Jetzt bewerten
            </a>
          </div>
          <p style="font-size: 0.875rem; color: #6b7280;">
            ${ownerName} und das ganze Team von ${businessName}
          </p>
        </div>
      </body>
    </html>
  `;
}

export function isMessagingConfigured(): { sms: boolean; email: boolean } {
  return { sms: !!getTwilio(), email: !!getResend() };
}
