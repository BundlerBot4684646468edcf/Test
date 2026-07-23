import express from 'express';
import { reviewRequests, customers, businesses } from '../db';

// Industry "conversion window": the ask goes out N hours after the visit,
// while satisfaction is at its peak. For customers served longer ago than
// that (e.g. past-customer imports), send at the next queue run instead.
const DEFAULT_SEND_DELAY_HOURS = 3;

export function computeScheduledAt(
  servedAtISO: string,
  sendDelayHours: number | null
): string {
  const delayMs =
    (sendDelayHours ?? DEFAULT_SEND_DELAY_HOURS) * 60 * 60 * 1000;
  const ideal = new Date(servedAtISO).getTime() + delayMs;
  return new Date(Math.max(ideal, Date.now())).toISOString();
}

const router = express.Router({ mergeParams: true });

// Attach the customer object onto a review request (frontend expects it).
function withCustomer(rr: any) {
  return { ...rr, customer: customers.get(rr.customerId) || null };
}

// GET /api/businesses/:businessId/review-requests
router.get('/', async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const status = req.query.status as string | undefined;
    const skip = parseInt((req.query.skip as string) || '0', 10);
    const take = parseInt((req.query.take as string) || '20', 10);

    const list = reviewRequests
      .list(businessId, { status, skip, take })
      .map(withCustomer);
    const total = reviewRequests.count(businessId, status);

    res.json({ requests: list, total });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to list review requests' });
  }
});

// POST /api/businesses/:businessId/review-requests — Create request
router.post('/', async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const { customerId, channel = 'sms' } = req.body;

    if (!['sms', 'whatsapp', 'email'].includes(channel)) {
      return res
        .status(400)
        .json({ error: 'channel must be sms, whatsapp or email' });
    }

    const customer = customers.get(customerId);
    if (!customer || customer.businessId !== businessId) {
      return res.status(404).json({ error: 'Customer not found' });
    }
    if (customer.optOut) {
      return res
        .status(400)
        .json({ error: 'Customer has opted out from notifications' });
    }

    const existing = reviewRequests.findActiveForCustomer(customerId);
    if (existing) {
      return res.status(400).json({ error: 'Request already exists' });
    }

    const business = businesses.get(businessId);
    const request = reviewRequests.create({
      customerId,
      businessId,
      channel,
      status: 'queued',
      scheduledAt: computeScheduledAt(
        customer.servedAt,
        business?.sendDelayHours ?? null
      ),
    });

    res.status(201).json(withCustomer(request));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create review request' });
  }
});

// PATCH /api/businesses/:businessId/review-requests/:requestId/opt-out
router.patch('/:requestId/opt-out', async (req, res) => {
  try {
    const { businessId, requestId } = req.params as Record<string, string>;

    const request = reviewRequests.get(requestId);
    if (!request || request.businessId !== businessId) {
      return res.status(404).json({ error: 'Request not found' });
    }

    customers.update(request.customerId, { optOut: 1 });
    reviewRequests.optOutForCustomer(request.customerId);

    res.json({ success: true });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to opt out' });
  }
});

export default router;
