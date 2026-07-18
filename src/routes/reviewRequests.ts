import express from 'express';
import prisma from '../db';

const router = express.Router({ mergeParams: true });

// GET /api/businesses/:businessId/review-requests
router.get('/', async (req, res) => {
  try {
    const { businessId } = req.params;
    const { status, skip = 0, take = 20 } = req.query;

    const where: any = { businessId };
    if (status) where.status = status;

    const requests = await prisma.reviewRequest.findMany({
      where,
      skip: parseInt(skip as string),
      take: parseInt(take as string),
      include: {
        customer: true,
      },
      orderBy: { createdAt: 'desc' },
    });

    const total = await prisma.reviewRequest.count({ where });

    res.json({ requests, total });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to list review requests' });
  }
});

// POST /api/businesses/:businessId/review-requests — Create request
router.post('/', async (req, res) => {
  try {
    const { businessId } = req.params;
    const { customerId, channel = 'sms' } = req.body;

    const customer = await prisma.customer.findUnique({
      where: { id: customerId },
    });

    if (!customer || customer.businessId !== businessId) {
      return res.status(404).json({ error: 'Customer not found' });
    }

    if (customer.optOut) {
      return res
        .status(400)
        .json({ error: 'Customer has opted out from notifications' });
    }

    // Check for existing pending request
    const existing = await prisma.reviewRequest.findFirst({
      where: {
        customerId,
        status: { in: ['queued', 'sent', 'reminded'] },
      },
    });

    if (existing) {
      return res.status(400).json({ error: 'Request already exists' });
    }

    const request = await prisma.reviewRequest.create({
      data: {
        customerId,
        businessId,
        channel,
        status: 'queued',
      },
      include: { customer: true },
    });

    res.status(201).json(request);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create review request' });
  }
});

// PATCH /api/businesses/:businessId/review-requests/:requestId/opt-out
router.patch('/:requestId/opt-out', async (req, res) => {
  try {
    const { businessId, requestId } = req.params;

    const request = await prisma.reviewRequest.findUnique({
      where: { id: requestId },
      include: { customer: true },
    });

    if (!request || request.businessId !== businessId) {
      return res.status(404).json({ error: 'Request not found' });
    }

    // Mark customer as opted out
    await prisma.customer.update({
      where: { id: request.customerId },
      data: { optOut: true },
    });

    // Mark all pending requests as opted out
    await prisma.reviewRequest.updateMany({
      where: {
        customerId: request.customerId,
        status: { in: ['queued', 'sent'] },
      },
      data: { status: 'opted_out' },
    });

    res.json({ success: true });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to opt out' });
  }
});

export default router;
