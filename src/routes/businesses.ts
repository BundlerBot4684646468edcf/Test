import { Router } from 'express';
import { businesses, customers, reviewEvents } from '../db';
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

    const business = businesses.create({ name, ownerName, timezone });
    res.status(201).json(business);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create business' });
  }
});

// GET /api/businesses/:id — Get business details (+ recent customers & events)
router.get('/:id', async (req, res) => {
  try {
    const business = businesses.get(req.params.id);
    if (!business) {
      return res.status(404).json({ error: 'Business not found' });
    }

    res.json({
      ...business,
      customers: customers.list(business.id, 0, 10),
      reviewEvents: reviewEvents.listRecent(business.id, 10),
    });
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

    const business = businesses.get(req.params.id);
    if (!business) {
      return res.status(404).json({ error: 'Business not found' });
    }

    const placeResult = await searchPlace(businessName, address);
    const reviewLink = generateReviewLink(placeResult.placeId);

    const updated = businesses.update(req.params.id, {
      googlePlaceId: placeResult.placeId,
      googleReviewLink: reviewLink,
    });

    res.json({
      success: true,
      place: placeResult,
      reviewLink,
      business: updated,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to find place' });
  }
});

// Nested routes
router.use('/:businessId/customers', customersRouter);
router.use('/:businessId/review-requests', reviewRequestsRouter);
router.use('/:businessId/photo', photosRouter);

export default router;
