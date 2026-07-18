import axios from 'axios';

const GOOGLE_API_KEY = process.env.GOOGLE_PLACES_API_KEY || 'mock_key';
const PLACES_API_BASE = 'https://maps.googleapis.com/maps/api/place';

export interface PlaceSearchResult {
  placeId: string;
  name: string;
  address: string;
  rating?: number;
  userRatingsTotal?: number;
}

export interface PlaceDetails {
  rating: number;
  userRatingsTotal: number;
  formattedAddress: string;
}

// Mock responses for development
const mockPlaceSearch = (): PlaceSearchResult => ({
  placeId: 'ChIJN1blFljjokMR5syQyewjZvQ',
  name: 'Mock Business',
  address: 'Via Roma 1, Bolzano, Italy',
  rating: 4.7,
  userRatingsTotal: 42,
});

const mockPlaceDetails = (): PlaceDetails => ({
  rating: 4.7,
  userRatingsTotal: 42,
  formattedAddress: 'Via Roma 1, Bolzano, 39100, Italy',
});

export async function searchPlace(
  businessName: string,
  address: string
): Promise<PlaceSearchResult> {
  if (!GOOGLE_API_KEY || GOOGLE_API_KEY === 'mock_key') {
    console.log(
      `[MOCK] Searching for place: "${businessName}" at "${address}"`
    );
    return mockPlaceSearch();
  }

  try {
    const query = `${businessName} ${address}`;
    const response = await axios.get(
      `${PLACES_API_BASE}/textsearch/json`,
      {
        params: {
          query,
          key: GOOGLE_API_KEY,
        },
      }
    );

    if (!response.data.results || response.data.results.length === 0) {
      throw new Error('Place not found');
    }

    const place = response.data.results[0];
    return {
      placeId: place.place_id,
      name: place.name,
      address: place.formatted_address,
      rating: place.rating,
      userRatingsTotal: place.user_ratings_total,
    };
  } catch (error) {
    console.error('Error searching place:', error);
    throw error;
  }
}

export async function getPlaceDetails(placeId: string): Promise<PlaceDetails> {
  if (!GOOGLE_API_KEY || GOOGLE_API_KEY === 'mock_key') {
    console.log(`[MOCK] Getting details for place: ${placeId}`);
    return mockPlaceDetails();
  }

  try {
    const response = await axios.get(`${PLACES_API_BASE}/details/json`, {
      params: {
        place_id: placeId,
        fields: 'rating,user_ratings_total,formatted_address',
        key: GOOGLE_API_KEY,
      },
    });

    const result = response.data.result;
    return {
      rating: result.rating || 0,
      userRatingsTotal: result.user_ratings_total || 0,
      formattedAddress: result.formatted_address || '',
    };
  } catch (error) {
    console.error('Error getting place details:', error);
    throw error;
  }
}

export function generateReviewLink(placeId: string): string {
  return `https://search.google.com/local/writereview?placeid=${placeId}`;
}
