import { businesses, reviewEvents } from '../db';
import { getPlaceDetails } from './googlePlaces';

export async function updateReviewMetrics() {
  console.log(`[METRICS] Updating review metrics at ${new Date().toISOString()}`);

  try {
    const list = businesses.listWithPlaceId();

    for (const business of list) {
      if (!business.googlePlaceId) continue;

      try {
        const details = await getPlaceDetails(business.googlePlaceId);

        const previous = reviewEvents.latest(business.id);
        const newReviewCount = previous
          ? Math.max(0, details.userRatingsTotal - previous.totalReviews)
          : 0;

        reviewEvents.create({
          businessId: business.id,
          newReviewCount,
          avgRating: details.rating,
          totalReviews: details.userRatingsTotal,
          checkedAt: new Date().toISOString(),
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
