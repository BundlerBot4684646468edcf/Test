import prisma from '../db';
import { getPlaceDetails } from './googlePlaces';

export async function updateReviewMetrics() {
  console.log(
    `[METRICS] Updating review metrics at ${new Date().toISOString()}`
  );

  try {
    const businesses = await prisma.business.findMany({
      where: { googlePlaceId: { not: null } },
    });

    for (const business of businesses) {
      if (!business.googlePlaceId) continue;

      try {
        const details = await getPlaceDetails(business.googlePlaceId);

        // Get previous review count
        const previousEvent = await prisma.reviewEvent.findFirst({
          where: { businessId: business.id },
          orderBy: { checkedAt: 'desc' },
        });

        const newReviewCount = previousEvent
          ? Math.max(0, details.userRatingsTotal - previousEvent.totalReviews)
          : 0;

        // Create new event
        await prisma.reviewEvent.create({
          data: {
            businessId: business.id,
            newReviewCount,
            avgRating: details.rating,
            totalReviews: details.userRatingsTotal,
            checkedAt: new Date(),
          },
        });

        console.log(
          `[METRICS] ${business.name}: ${newReviewCount} new reviews, rating: ${details.rating}`
        );
      } catch (error) {
        console.error(
          `[METRICS ERROR] Failed to update metrics for ${business.name}:`,
          error
        );
      }
    }
  } catch (error) {
    console.error('[METRICS ERROR]', error);
  }
}
