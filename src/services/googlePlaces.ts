import axios from 'axios';

const PLACES_API_BASE = 'https://maps.googleapis.com/maps/api/place';

// Read the key lazily (inside functions) so it is available regardless of
// when the .env file finishes loading relative to module import order.
function getApiKey(): string {
  return process.env.GOOGLE_PLACES_API_KEY || '';
}

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

export async function searchPlace(
  businessName: string,
  address: string
): Promise<PlaceSearchResult> {
  const apiKey = getApiKey();
  if (!apiKey) {
    throw new Error(
      'Google Places API key not configured. Set GOOGLE_PLACES_API_KEY in .env'
    );
  }

  try {
    const query = `${businessName} ${address}`;
    const response = await axios.get(`${PLACES_API_BASE}/textsearch/json`, {
      params: { query, key: apiKey },
    });

    if (!response.data.results || response.data.results.length === 0) {
      throw new Error('Place not found');
    }

    const place = response.data.results[0];
    console.log(`✅ Found place: ${place.name}`);
    return {
      placeId: place.place_id,
      name: place.name,
      address: place.formatted_address,
      rating: place.rating,
      userRatingsTotal: place.user_ratings_total,
    };
  } catch (error) {
    console.error('❌ Place search error:', error);
    throw error;
  }
}

export async function getPlaceDetails(placeId: string): Promise<PlaceDetails> {
  const apiKey = getApiKey();
  if (!apiKey) {
    throw new Error(
      'Google Places API key not configured. Set GOOGLE_PLACES_API_KEY in .env'
    );
  }

  try {
    const response = await axios.get(`${PLACES_API_BASE}/details/json`, {
      params: {
        place_id: placeId,
        fields: 'rating,user_ratings_total,formatted_address',
        key: apiKey,
      },
    });

    const result = response.data.result;
    console.log(`✅ Got place details: ${result.user_ratings_total} reviews`);
    return {
      rating: result.rating || 0,
      userRatingsTotal: result.user_ratings_total || 0,
      formattedAddress: result.formatted_address || '',
    };
  } catch (error) {
    console.error('❌ Place details error:', error);
    throw error;
  }
}

export function generateReviewLink(placeId: string): string {
  return `https://search.google.com/local/writereview?placeid=${placeId}`;
}

export function isGooglePlacesConfigured(): boolean {
  return !!getApiKey();
}
