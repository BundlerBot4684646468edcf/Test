import { reviewRequests, customers, businesses } from '../db';
import {
  sendSMS,
  sendEmail,
  buildReviewRequestSMS,
  buildReviewRequestHTML,
} from './messaging';

const REMINDER_WAIT_DAYS = 3;

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
          mediaUrl: business.ownerPhotoUrl || undefined,
        });
        sent = result.success;
      } else if (request.channel === 'email' && customer.email) {
        const html = buildReviewRequestHTML(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName,
          business.ownerPhotoUrl || undefined
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
