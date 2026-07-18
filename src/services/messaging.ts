import { Resend } from 'resend';

const TWILIO_ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID || 'mock';
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN || 'mock';
const TWILIO_PHONE = process.env.TWILIO_PHONE_NUMBER || '+1234567890';
const RESEND_API_KEY = process.env.RESEND_API_KEY || 'mock';

const resend = new Resend(RESEND_API_KEY);

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

// Mock responses
const mockSmsSend = async (payload: SMSPayload) => {
  console.log(`[MOCK SMS] To: ${payload.toPhone}`);
  console.log(`[MOCK SMS] Message: ${payload.message}`);
  if (payload.mediaUrl) {
    console.log(`[MOCK SMS] MediaUrl: ${payload.mediaUrl}`);
  }
  return { success: true, messageId: `mock-sms-${Date.now()}` };
};

const mockEmailSend = async (payload: EmailPayload) => {
  console.log(`[MOCK EMAIL] To: ${payload.toEmail}`);
  console.log(`[MOCK EMAIL] Subject: ${payload.subject}`);
  return { id: `mock-email-${Date.now()}` };
};

export async function sendSMS(payload: SMSPayload): Promise<{
  success: boolean;
  messageId?: string;
  error?: string;
}> {
  if (TWILIO_ACCOUNT_SID === 'mock') {
    return mockSmsSend(payload);
  }

  try {
    // Real Twilio would go here
    // For now, return mock response
    return mockSmsSend(payload);
  } catch (error) {
    console.error('SMS error:', error);
    return { success: false, error: String(error) };
  }
}

export async function sendEmail(payload: EmailPayload): Promise<{
  success: boolean;
  messageId?: string;
  error?: string;
}> {
  if (RESEND_API_KEY === 'mock') {
    return mockEmailSend(payload);
  }

  try {
    const result = await resend.emails.send({
      from: `${payload.fromName || 'Mundpost'} <noreply@resend.dev>`,
      to: payload.toEmail,
      subject: payload.subject,
      html: payload.html,
    });

    if (result.error) {
      return { success: false, error: result.error.message };
    }

    return { success: true, messageId: result.data?.id };
  } catch (error) {
    console.error('Email error:', error);
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
  const baseMessage = `Hallo ${firstName},\n\nIch bin's, ${ownerName} von ${businessName}. Danke dass du bei uns warst! 🙏\n\nHättest du vielleicht 30 Sekunden Zeit für eine kurze Google-Bewertung? Das bedeutet uns wirklich viel.\n\n${reviewLink}`;
  return baseMessage;
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
