import { promises as fsp } from 'fs';
import { reviewRequests, customers, businesses, Business } from '../db';
import {
  sendSMS,
  sendEmail,
  buildReviewRequestSMS,
  buildReviewRequestHTML,
} from './messaging';
import { uploadFile, localPathForUrl } from './fileStorage';
import {
  personalizePhoto,
  DEFAULT_SIGN_BOX,
} from './photoPersonalization';

const REMINDER_WAIT_DAYS = 3;

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

      let sent = false;

      // Per-customer personalized photo (name written on the sign)
      const personalPhotoUrl = await photoUrlFor(business, customer.firstName);

      if (request.channel === 'sms' && customer.phone) {
        const message = buildReviewRequestSMS(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName
        );
        const result = await sendSMS({
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
          personalPhotoUrl
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

    // Reminders: 3+ days after send, no reminder yet
    const cutoff = new Date(
      Date.now() - REMINDER_WAIT_DAYS * 24 * 60 * 60 * 1000
    ).toISOString();
    const reminders = reviewRequests.listReminderCandidates(cutoff, 50);

    for (const request of reminders) {
      const customer = customers.get(request.customerId);
      const business = businesses.get(request.businessId);
      if (!customer || !business || customer.optOut) continue;

      const reminderMessage = `Kurze Erinnerung: Hättest du 30 Sekunden für die Google-Bewertung? ${business.googleReviewLink}`;

      if (request.channel === 'sms' && customer.phone) {
        await sendSMS({ toPhone: customer.phone, message: reminderMessage });
      } else if (request.channel === 'email' && customer.email) {
        await sendEmail({
          toEmail: customer.email,
          subject: 'Kurze Erinnerung: Google-Bewertung',
          html: `<p>${reminderMessage}</p>`,
          fromName: business.ownerName,
        });
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
