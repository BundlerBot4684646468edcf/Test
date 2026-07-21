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
  const greeting = servedBy && servedBy.trim() && servedBy.trim() !== ownerName.trim()
    ? `${servedBy.trim()}`
    : ownerName;
  return `Hallo ${firstName},\n\nes ist ${greeting} von ${businessName}! 💈 ${visitLine(servedBy, ownerName)}\n\nKannst du mir einen großen Gefallen tun? Wenn du kurz Zeit hast, würde ich mich riesig freuen, wenn du uns auf Google bewertest — das bedeutet uns echt viel! 🙏\n\n${reviewLink}`;
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
    ? `<img src="${ownerPhotoUrl}" alt="${ownerName}" style="width: 100%; max-width: 480px; border-radius: 12px; margin-bottom: 1.5rem;" />`
    : '';

  const senderName = servedBy && servedBy.trim() && servedBy.trim() !== ownerName.trim()
    ? servedBy.trim()
    : ownerName;

  return `
    <html>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f9fafb;">
        <div style="max-width: 500px; margin: 0 auto; padding: 2rem;">
          <div style="background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            ${photoSection}
            <p style="margin: 0 0 1rem 0; font-size: 0.95rem; color: #6b7280;">
              ❤️ <strong>Hallo ${firstName}</strong>
            </p>
            <p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; line-height: 1.6; color: #1f2937;">
              es ist ${senderName} von ${businessName}! 💈<br>
              ${visitLine(servedBy, ownerName)}
            </p>
            <p style="margin: 0 0 1.5rem 0; font-size: 0.95rem; line-height: 1.6; color: #1f2937;">
              Kannst du mir einen großen Gefallen tun? Wenn du kurz Zeit hast, würde ich mich riesig freuen, wenn du uns auf Google bewertest — das bedeutet uns echt viel! 🙏
            </p>
            <div style="margin: 2rem 0; text-align: center;">
              <a href="${reviewLink}" style="background-color: #d63031; color: white; padding: 0.875rem 2rem; border-radius: 6px; text-decoration: none; display: inline-block; font-weight: 600; font-size: 0.95rem;">
                Google-Bewertung schreiben
              </a>
            </div>
            <p style="margin: 1.5rem 0 0 0; padding-top: 1.5rem; border-top: 1px solid #e5e7eb; font-size: 0.85rem; color: #9ca3af;">
              ${senderName} und das Team von ${businessName}
            </p>
          </div>
        </div>
      </body>
    </html>
  `;
}

export function isMessagingConfigured(): { sms: boolean; email: boolean } {
  return { sms: !!getTwilio(), email: !!getResend() };
}
