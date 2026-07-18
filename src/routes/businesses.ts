import express, { Router } from 'express';
import prisma from '../db';
import { searchPlace, generateReviewLink } from '../services/googlePlaces';
import customersRouter from './customers';
import photosRouter from './photos';
import reviewRequestsRouter from './reviewRequests';

const router = Router();

// POST /api/businesses — Create a new business
router.post('/', async (req, res) => {
  try {
    const { name, ownerName, timezone = 'Europe/Rome' } = req.body;

    if (!name || !ownerName) {
      return res
        .status(400)
        .json({ error: 'name and ownerName are required' });
    }

    const business = await prisma.business.create({
      data: { name, ownerName, timezone },
    });

    res.status(201).json(business);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create business' });
  }
});

// GET /api/businesses/:id — Get business details
router.get('/:id', async (req, res) => {
  try {
    const business = await prisma.business.findUnique({
      where: { id: req.params.id },
      include: {
        customers: { take: 10 },
        reviewEvents: { take: 10, orderBy: { checkedAt: 'desc' } },
      },
    });

    if (!business) {
      return res.status(404).json({ error: 'Business not found' });
    }

    res.json(business);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to get business' });
  }
});

// POST /api/businesses/:id/find-place — Search for Google Place
router.post('/:id/find-place', async (req, res) => {
  try {
    const { businessName, address } = req.body;

    if (!businessName || !address) {
      return res
        .status(400)
        .json({ error: 'businessName and address are required' });
    }

    const placeResult = await searchPlace(businessName, address);
    const reviewLink = generateReviewLink(placeResult.placeId);

    // Save to business
    const business = await prisma.business.update({
      where: { id: req.params.id },
      data: {
        googlePlaceId: placeResult.placeId,
        googleReviewLink: reviewLink,
      },
    });

    res.json({
      success: true,
      place: placeResult,
      reviewLink,
      business,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to find place' });
  }
});

// Nested routes
router.use('/:businessId/customers', customersRouter);
router.use('/:businessId/review-requests', reviewRequestsRouter);
router.use('/:id/photo', photosRouter);

export default router;
