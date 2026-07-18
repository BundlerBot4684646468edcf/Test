import prisma from '../db';
import { sendSMS, sendEmail, buildReviewRequestSMS, buildReviewRequestHTML } from './messaging';

const REMINDER_WAIT_DAYS = 3;

export async function processReviewQueue() {
  console.log(`[CRON] Processing review queue at ${new Date().toISOString()}`);

  try {
    // Get all pending review requests
    const pendingRequests = await prisma.reviewRequest.findMany({
      where: {
        status: 'queued',
        createdAt: { lte: new Date() }, // Immediate send for now
      },
      include: {
        customer: true,
        business: true,
      },
      take: 100, // Process in batches
    });

    console.log(`Found ${pendingRequests.length} pending review requests`);

    for (const request of pendingRequests) {
      const { customer, business, channel } = request;

      // Check rate limiting
      const sentToday = await prisma.reviewRequest.count({
        where: {
          businessId: business.id,
          status: { in: ['sent', 'reminded', 'reviewed'] },
          sentAt: {
            gte: new Date(new Date().setHours(0, 0, 0, 0)),
            lte: new Date(new Date().setHours(23, 59, 59, 999)),
          },
        },
      });

      if (sentToday >= business.dailyBatchLimit) {
        console.log(
          `[RATE LIMIT] Business ${business.id} reached daily limit`
        );
        continue;
      }

      // Skip if customer has opted out
      if (customer.optOut) {
        await prisma.reviewRequest.update({
          where: { id: request.id },
          data: { status: 'opted_out' },
        });
        continue;
      }

      // Send message
      let sendResult: { success: boolean };

      if (channel === 'sms' && customer.phone) {
        const message = buildReviewRequestSMS(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName,
          business.ownerPhotoUrl || undefined
        );

        sendResult = await sendSMS({
          toPhone: customer.phone,
          message,
          mediaUrl: business.ownerPhotoUrl,
        });
      } else if (channel === 'email' && customer.email) {
        const html = buildReviewRequestHTML(
          customer.firstName,
          business.name,
          business.googleReviewLink || 'https://google.com',
          business.ownerName,
          business.ownerPhotoUrl
        );

        sendResult = await sendEmail({
          toEmail: customer.email,
          subject: `Eine kurze Frage von ${business.ownerName}`,
          html,
          fromName: business.ownerName,
        });
      } else {
        console.warn(`No valid contact info for customer ${customer.id}`);
        continue;
      }

      if (sendResult.success) {
        await prisma.reviewRequest.update({
          where: { id: request.id },
          data: {
            status: 'sent',
            sentAt: new Date(),
          },
        });
      } else {
        console.error(`Failed to send to ${customer.id}`);
      }
    }

    // Handle reminders (3+ days after first send)
    const reminderCandidates = await prisma.reviewRequest.findMany({
      where: {
        status: 'sent',
        sentAt: {
          lte: new Date(Date.now() - REMINDER_WAIT_DAYS * 24 * 60 * 60 * 1000),
        },
        remindedAt: null,
      },
      include: {
        customer: true,
        business: true,
      },
      take: 50,
    });

    for (const request of reminderCandidates) {
      const { customer, business, channel } = request;

      if (customer.optOut) continue;

      const reminderMessage = `Kurze Erinnerung: Hättest du 30 Sekunden für die Google-Bewertung? ${business.googleReviewLink}`;

      if (channel === 'sms' && customer.phone) {
        await sendSMS({
          toPhone: customer.phone,
          message: reminderMessage,
        });
      } else if (channel === 'email' && customer.email) {
        await sendEmail({
          toEmail: customer.email,
          subject: `Kurze Erinnerung: Google-Bewertung`,
          html: `<p>${reminderMessage}</p>`,
          fromName: business.ownerName,
        });
      }

      await prisma.reviewRequest.update({
        where: { id: request.id },
        data: { remindedAt: new Date(), status: 'reminded' },
      });
    }

    console.log('[CRON] Review queue processing complete');
  } catch (error) {
    console.error('[CRON ERROR]', error);
  }
}
