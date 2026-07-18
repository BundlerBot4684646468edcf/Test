import { Resend } from 'resend';
import twilio from 'twilio';

const TWILIO_ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID || '';
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN || '';
const TWILIO_PHONE = process.env.TWILIO_PHONE_NUMBER || '';
const RESEND_API_KEY = process.env.RESEND_API_KEY || '';

let twilioClient: any = null;
let resendClient: Resend | null = null;

// Initialize Twilio only if credentials exist
if (TWILIO_ACCOUNT_SID && TWILIO_AUTH_TOKEN) {
  twilioClient = twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);
}

// Initialize Resend only if API key exists
if (RESEND_API_KEY) {
  resendClient = new Resend(RESEND_API_KEY);
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
  if (!twilioClient) {
    console.warn(
      '[SMS] Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER'
    );
    return {
      success: false,
      error: 'Twilio not configured',
    };
  }

  try {
    const messageData: any = {
      body: payload.message,
      from: TWILIO_PHONE,
      to: payload.toPhone,
    };

    // Add media if photo URL provided (MMS)
    if (payload.mediaUrl) {
      messageData.mediaUrl = [payload.mediaUrl];
    }

    const message = await twilioClient.messages.create(messageData);

    console.log(`✅ SMS sent: ${message.sid} to ${payload.toPhone}`);
    return { success: true, messageId: message.sid };
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
  if (!resendClient) {
    console.warn('[EMAIL] Resend not configured. Set RESEND_API_KEY');
    return {
      success: false,
      error: 'Resend not configured',
    };
  }

  try {
    const result = await resendClient.emails.send({
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
export function buildReviewRequestSMS(
  firstName: string,
  businessName: string,
  reviewLink: string,
  ownerName: string,
  ownerPhotoUrl?: string
): string {
  return `Hallo ${firstName},\n\nIch bin's, ${ownerName} von ${businessName}. Danke dass du bei uns warst! 🙏\n\nHättest du vielleicht 30 Sekunden Zeit für eine kurze Google-Bewertung? Das bedeutet uns wirklich viel.\n\n${reviewLink}`;
}

export function buildReviewRequestHTML(
  firstName: string,
  businessName: string,
  reviewLink: string,
  ownerName: string,
  ownerPhotoUrl?: string
): string {
  const photoSection = ownerPhotoUrl
    ? `<img src="${ownerPhotoUrl}" alt="${ownerName}" style="width: 100px; height: 100px; border-radius: 50%; margin-bottom: 1rem;" />`
    : '';

  return `
    <html>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="max-width: 500px; margin: 0 auto; padding: 2rem;">
          ${photoSection}
          <h2>Hallo ${firstName}!</h2>
          <p>Ich bin's, ${ownerName} von ${businessName}. Danke dass du bei uns warst! 🙏</p>
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

export function isMessagingConfigured(): {
  sms: boolean;
  email: boolean;
} {
  return {
    sms: !!twilioClient,
    email: !!resendClient,
  };
}
