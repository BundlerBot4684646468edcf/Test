// Cloudflare Worker for Amadeus API Token Authentication
// Handles CORS and securely fetches OAuth tokens

export default {
  async fetch(request, env) {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // Validate environment variables
      if (!env.AMADEUS_CLIENT_ID || !env.AMADEUS_CLIENT_SECRET) {
        throw new Error('Missing AMADEUS_CLIENT_ID or AMADEUS_CLIENT_SECRET environment variables');
      }

      console.log('Requesting Amadeus OAuth token...');

      // Fetch Amadeus OAuth token
      const tokenRes = await fetch('https://api.amadeus.com/v1/security/oauth2/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
          grant_type: 'client_credentials',
          client_id: env.AMADEUS_CLIENT_ID,
          client_secret: env.AMADEUS_CLIENT_SECRET
        }).toString()
      });

      // Check response status
      if (!tokenRes.ok) {
        const errorText = await tokenRes.text();
        console.error('Amadeus Auth error:', tokenRes.status, errorText);
        throw new Error(`Amadeus Auth failed: ${tokenRes.status} ${tokenRes.statusText}`);
      }

      // Parse JSON response
      const tokenData = await tokenRes.json();

      // Validate response has access_token
      if (!tokenData.access_token) {
        console.error('No access_token in response:', tokenData);
        throw new Error('Amadeus Auth response missing access_token');
      }

      console.log('✅ Amadeus token acquired successfully');

      // Return token data
      return new Response(JSON.stringify({
        access_token: tokenData.access_token,
        expires_in: tokenData.expires_in,
        token_type: tokenData.token_type
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store' // Don't cache sensitive tokens
        }
      });

    } catch (error) {
      console.error('Worker error:', error.message);

      return new Response(JSON.stringify({
        error: error.message,
        timestamp: new Date().toISOString()
      }), {
        status: error.message.includes('Missing') ? 500 : 502,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        }
      });
    }
  }
}
