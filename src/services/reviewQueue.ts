import { promises as fsp } from 'fs';
import { reviewRequests, customers, businesses, Business } from '../db';
import {
  sendSMS,
  sendEmail,
  sendWhatsApp,
  buildReviewRequestSMS,
  buildReviewRequestHTML,
} from './messaging';
import { uploadFile, localPathForUrl } from './fileStorage';
import {
  personalizePhoto,
  DEFAULT_SIGN_BOX,
} from './photoPersonalization';

// Follow-up fires this many hours after the first send with no review,
// unless the business overrides it (reminderDelayHours).
const DEFAULT_REMINDER_DELAY_HOURS = 24;

// EU marketing rule of thumb: only send between these local hours. A review
// ask that buzzes someone's phone at 3am is both illegal-feeling and ignored.
const SEND_WINDOW_START = 8;
const SEND_WINDOW_END = 20;

/** Current hour (0–23) in the business's timezone. */
function localHour(timezone: string): number {
  try {
    const s = new Intl.DateTimeFormat('en-GB', {
      timeZone: timezone,
      hour: 'numeric',
      hour12: false,
    }).format(new Date());
    return parseInt(s, 10) % 24;
  } catch {
    return new Date().getHours();
  }
}

function withinSendingWindow(timezone: string): boolean {
  const h = localHour(timezone);
  return h >= SEND_WINDOW_START && h < SEND_WINDOW_END;
}

/** Public opt-out + privacy links, built from PUBLIC_BASE_URL. */
function publicLinks(customerId: string): { unsubscribeUrl?: string; privacyUrl?: string } {
  const base = (process.env.PUBLIC_BASE_URL || '').replace(/\/$/, '');
  if (!base) return {};
  return {
    unsubscribeUrl: `${base}/u/${customerId}`,
    privacyUrl: `${base}/datenschutz`,
  };
}

/**
 * If the business has an owner photo (with a blank sign) stored locally,
 * render the customer's name onto it and store the result, returning its URL.
 * Falls back to the plain owner photo on any problem.
 */
async function photoUrlFor(
  business: Business,
  firstName: string
): Promise<string | undefined> {
  if (!business.ownerPhotoUrl) return undefined;
  try {
    const filePath = localPathForUrl(business.ownerPhotoUrl);
    if (!filePath) return business.ownerPhotoUrl; // hosted elsewhere, use as-is

    const base = await fsp.readFile(filePath);
    const box =
      business.signX != null
        ? {
            x: business.signX,
            y: business.signY ?? 0.6,
            width: business.signWidth ?? 0.5,
            height: business.signHeight ?? 0.2,
            rotation: business.signRotation ?? 0,
          }
        : DEFAULT_SIGN_BOX;

    const png = await personalizePhoto(base, firstName, box);
    return await uploadFile(
      png,
      `${firstName.toLowerCase()}.png`,
      `personalized/${business.id}`
    );
  } catch (error) {
    console.warn('[PHOTO] Personalization failed, using plain photo:', error);
    return business.ownerPhotoUrl;
  }
}

function startOfTodayISO(): string {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.toISOString();
}

export async function processReviewQueue() {
  console.log(`[CRON] Processing review queue at ${new Date().toISOString()}`);

  try {
    const due = reviewRequests.listDue(100);
    console.log(`Found ${due.length} due review requests`);

    for (const request of due) {
      const customer = customers.get(request.customerId);
      const business = businesses.get(request.businessId);
      if (!customer || !business) continue;

      // Opt-out guard
      if (customer.optOut) {
        reviewRequests.update(request.id, { status: 'opted_out' });
        continue;
      }

      // Rate limit: never exceed dailyBatchLimit sends per business per day
      const sentToday = reviewRequests.countSentToday(
        business.id,
        startOfTodayISO()
      );
      if (sentToday >= business.dailyBatchLimit) {
        console.log(`[RATE LIMIT] Business ${business.id} reached daily limit`);
        continue;
      }

      // Only send within polite local hours (per-business timezone).
      if (!withinSendingWindow(business.timezone)) {
        console.log(
          `[SEND WINDOW] Outside ${SEND_WINDOW_START}-${SEND_WINDOW_END}h in ${business.timezone}, deferring ${request.id}`
        );
        continue;
      }

      let sent = false;

      // Per-customer personalized photo (name written on the sign)
      const personalPhotoUrl = await photoUrlFor(business, customer.firstName);
      const { unsubscribeUrl, privacyUrl } = publicLinks(customer.id);

      if (request.channel === 'sms' && customer.phone) {
        const message = buildReviewRequestSMS(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName,
          customer.servedBy,
          unsubscribeUrl
        );
        const result = await sendSMS({
          toPhone: customer.phone,
          message,
          mediaUrl: personalPhotoUrl,
        });
        sent = result.success;
      } else if (request.channel === 'whatsapp' && customer.phone) {
        // WhatsApp shows the photo inline in the chat — even in Italy.
        const message = buildReviewRequestSMS(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName,
          customer.servedBy,
          unsubscribeUrl
        );
        const result = await sendWhatsApp({
          toPhone: customer.phone,
          message,
          mediaUrl: personalPhotoUrl,
        });
        sent = result.success;
      } else if (request.channel === 'email' && customer.email) {
        const html = buildReviewRequestHTML(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName,
          personalPhotoUrl,
          customer.servedBy,
          unsubscribeUrl,
          privacyUrl
        );
        const result = await sendEmail({
          toEmail: customer.email,
          subject: `Eine kurze Frage von ${business.ownerName}`,
          html,
          fromName: business.ownerName,
        });
        sent = result.success;
      } else {
        console.warn(`No valid contact info for customer ${customer.id}`);
        continue;
      }

      if (sent) {
        reviewRequests.update(request.id, {
          status: 'sent',
          sentAt: new Date().toISOString(),
        });
      } else {
        console.error(`Failed to send to ${customer.id}`);
      }
    }

    // Follow-ups: default 24h after the send (per-business override),
    // switching to email when the first ask went out by SMS — a second SMS
    // feels pushy, a single friendly email does not.
    const candidates = reviewRequests.listReminderCandidates(
      new Date().toISOString(),
      50
    );

    for (const request of candidates) {
      const customer = customers.get(request.customerId);
      const business = businesses.get(request.businessId);
      if (!customer || !business || customer.optOut || !request.sentAt) continue;

      const delayMs =
        (business.reminderDelayHours ?? DEFAULT_REMINDER_DELAY_HOURS) *
        60 * 60 * 1000;
      if (Date.now() - new Date(request.sentAt).getTime() < delayMs) continue;

      // Respect the same polite-hours window for follow-ups.
      if (!withinSendingWindow(business.timezone)) continue;

      const { unsubscribeUrl, privacyUrl } = publicLinks(customer.id);
      const reminderMessage = `Hallo ${customer.firstName}, kurze Erinnerung von ${business.ownerName}: Hättest du 30 Sekunden für eine Google-Bewertung? Das würde uns riesig freuen. ${business.googleReviewLink}${
        unsubscribeUrl ? `\n\nAbmelden: ${unsubscribeUrl}` : ''
      }`;

      // Channel switch: SMS/WhatsApp first, email as the follow-up (if we
      // have one) — a second ping on the same channel feels pushy.
      const useEmail = !!customer.email;

      if (useEmail) {
        const footer = unsubscribeUrl
          ? `<p style="font-size:0.75rem;color:#9ca3af;margin-top:1.5rem;">
               <a href="${unsubscribeUrl}" style="color:#0066cc;">Abmelden</a>${
                 privacyUrl ? ` · <a href="${privacyUrl}" style="color:#0066cc;">Datenschutz</a>` : ''
               }
             </p>`
          : '';
        await sendEmail({
          toEmail: customer.email!,
          subject: `Kurze Erinnerung von ${business.ownerName} 🙏`,
          html: `<p>${reminderMessage}</p>${footer}`,
          fromName: business.ownerName,
        });
      } else if (customer.phone) {
        await sendSMS({ toPhone: customer.phone, message: reminderMessage });
      } else {
        continue;
      }

      reviewRequests.update(request.id, {
        status: 'reminded',
        remindedAt: new Date().toISOString(),
      });
    }

    console.log('[CRON] Review queue processing complete');
  } catch (error) {
    console.error('[CRON ERROR]', error);
  }
}
