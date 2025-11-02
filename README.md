# 🏨 AI Hotel Sentiment Analysis Tool

Professional sentiment analysis system for hospitality industry, analyzing reviews from Booking.com, Google Reviews, TripAdvisor, and other platforms.

## Features

- **Multilingual Analysis** (DE/IT/EN)
- **LLM-powered** sentiment detection with OpenAI
- **ML clustering** to identify key operational themes
- **Time-weighted scoring** for freshness
- **Actionable insights** for hotel management
- **Export capabilities** (CSV)

## Architecture

### 1. Streamlit App (`app.py`)
Main sentiment analysis application with:
- Google Places API integration
- OpenAI GPT-4 analysis
- KMeans clustering
- Interactive visualizations

### 2. Cloudflare Worker (`worker.js`)
Serverless API for Amadeus authentication:
- Secure token fetching
- CORS handling
- Environment variable management

## Setup

### Prerequisites

- Python 3.9+
- Node.js (for Cloudflare Workers)
- Cloudflare account
- API Keys: OpenAI, Google Places, Amadeus

### Python App Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Add your API keys to .env
OPENAI_API_KEY=your_openai_key
GOOGLE_PLACES_API_KEY=your_google_key
```

### Run Streamlit App

```bash
streamlit run app.py
```

## Cloudflare Worker Deployment

### Local Development

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create .dev.vars from example
cp .dev.vars.example .dev.vars

# Add your Amadeus credentials to .dev.vars
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret

# Run locally
wrangler dev
```

### Production Deployment

```bash
# Set secrets (do this once)
wrangler secret put AMADEUS_CLIENT_ID
# Enter your client ID when prompted

wrangler secret put AMADEUS_CLIENT_SECRET
# Enter your client secret when prompted

# Deploy to Cloudflare
wrangler deploy
```

## Usage

### Analyze Hotel Reviews

1. Enter hotel name in the search box
2. Click "Fetch Reviews" to pull from Google Places
3. Optionally upload a CSV with additional reviews
4. Click "Run Analysis" to process

### CSV Format

Your CSV should include these columns:
- `date` (YYYY-MM-DD)
- `platform` (e.g., "Booking", "Google", "TripAdvisor")
- `language` (e.g., "DE", "IT", "EN")
- `rating` (1-10)
- `review_text` (the actual review content)

### Output Structure

```json
{
  "hotel_name": "Hotel Name",
  "city": "City",
  "overall_sentiment_score": 0.85,
  "review_clusters": {
    "Rooms": { "score": 0.87, "summary": "..." },
    "Service": { "score": 0.92, "summary": "..." },
    "Food": { "score": 0.68, "summary": "..." }
  },
  "key_issues": ["AC noise", "cold coffee"],
  "recommended_actions": ["..."]
}
```

## API Endpoints

### Amadeus Token Worker

**Endpoint:** `https://your-worker.workers.dev/`

**Method:** `POST` or `GET`

**Response:**
```json
{
  "access_token": "...",
  "expires_in": 1799,
  "token_type": "Bearer"
}
```

## Security Notes

⚠️ **Never commit credentials to git!**

- Use `.env` for Python secrets
- Use `.dev.vars` for local Worker development
- Use `wrangler secret` for production Worker secrets
- Both `.env` and `.dev.vars` are in `.gitignore`

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── worker.js               # Cloudflare Worker for Amadeus
├── wrangler.toml           # Worker configuration
├── requirements.txt        # Python dependencies
├── .env                    # Python environment variables (git-ignored)
├── .dev.vars               # Worker local env vars (git-ignored)
├── .dev.vars.example       # Template for local development
└── README.md              # This file
```

## Troubleshooting

### "Amadeus Auth response is not valid JSON"

- Ensure your credentials are correctly set in `.dev.vars` (local) or via `wrangler secret` (production)
- Check your Amadeus API credentials are valid
- Verify you're using the correct API endpoint

### "Missing GOOGLE_PLACES_API_KEY"

- Make sure `.env` file exists with `GOOGLE_PLACES_API_KEY=...`
- Restart Streamlit after modifying `.env`

### No reviews returned from Google

- Google Places API typically returns only ~5 recent reviews
- Upload a CSV for more comprehensive analysis

## License

MIT
