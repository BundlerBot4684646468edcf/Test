# 🚀 Cloudflare Worker Deployment Guide

## Prerequisites

- [Cloudflare account](https://dash.cloudflare.com/sign-up) (free tier works!)
- Node.js installed (v16 or higher)
- Your Amadeus API credentials

## Step 1: Install Node.js (if not installed)

Check if you have Node.js:
```bash
node --version
```

If not installed, download from: https://nodejs.org/ (use LTS version)

## Step 2: Install Wrangler CLI

Wrangler is Cloudflare's command-line tool for managing Workers.

```bash
npm install -g wrangler
```

Verify installation:
```bash
wrangler --version
```

## Step 3: Login to Cloudflare

```bash
wrangler login
```

This will:
- Open your browser
- Ask you to authorize Wrangler
- Save your credentials locally

## Step 4: Set Up Your Credentials

### Option A: Local Development

Create `.dev.vars` file for local testing:

```bash
# Copy the example file
cp .dev.vars.example .dev.vars
```

Edit `.dev.vars` and add your **REAL** Amadeus credentials:
```
AMADEUS_CLIENT_ID=your_actual_client_id_here
AMADEUS_CLIENT_SECRET=your_actual_client_secret_here
```

⚠️ **IMPORTANT**: `.dev.vars` is in `.gitignore` - never commit this file!

## Step 5: Test Locally

Run the worker on your local machine:

```bash
wrangler dev
```

You should see:
```
⛅️ wrangler 3.x.x
------------------
⬣ Listening on http://localhost:8787
```

**Test it:**
```bash
# In another terminal
curl http://localhost:8787
```

You should get a JSON response with an access token!

## Step 6: Deploy to Production

### 6.1: Set Production Secrets

Instead of `.dev.vars`, production uses encrypted secrets:

```bash
wrangler secret put AMADEUS_CLIENT_ID
```
When prompted, paste your Amadeus Client ID and press Enter.

```bash
wrangler secret put AMADEUS_CLIENT_SECRET
```
When prompted, paste your Amadeus Client Secret and press Enter.

### 6.2: Deploy

```bash
wrangler deploy
```

You'll see output like:
```
✨ Successfully published your Worker!
🌍 https://hotel-amadeus-auth.your-subdomain.workers.dev
```

**🎉 Your Worker is now live!** Copy that URL.

## Step 7: Test Production Deployment

```bash
curl https://hotel-amadeus-auth.your-subdomain.workers.dev
```

You should get:
```json
{
  "access_token": "...",
  "expires_in": 1799,
  "token_type": "Bearer"
}
```

## Troubleshooting

### "Authentication error"
- Make sure you ran `wrangler login`
- Try `wrangler logout` then `wrangler login` again

### "Missing AMADEUS_CLIENT_ID"
- Run `wrangler secret put AMADEUS_CLIENT_ID` again
- Make sure you pasted the correct value

### "Amadeus Auth failed: 401"
- Your Amadeus credentials are incorrect
- Get new credentials from: https://developers.amadeus.com/

### "Module not found"
- Make sure you're in the correct directory
- `worker.js` should be in the same folder as `wrangler.toml`

## Viewing Logs

See real-time logs:
```bash
wrangler tail
```

Then make requests to your Worker to see logs appear.

## Updating Your Worker

After making changes to `worker.js`:

```bash
wrangler deploy
```

That's it! Cloudflare will automatically update your live Worker.

## Deleting Your Worker

If you want to remove it:

```bash
wrangler delete
```

## Get Your Amadeus API Credentials

If you don't have them yet:

1. Go to: https://developers.amadeus.com/
2. Sign up / Log in
3. Go to "My Self-Service Workspace"
4. Create a new app
5. Copy the **Client ID** and **Client Secret**

## Cost

Cloudflare Workers **Free Tier**:
- 100,000 requests/day
- First 10ms of CPU time per request

For most hotel tools, this is **completely free**!

## Next Steps

Once deployed, you can:
- Use the Worker URL in your Python app
- Call it from your frontend
- Add it to your Streamlit dashboard

Example Python usage:
```python
import requests

worker_url = "https://hotel-amadeus-auth.your-subdomain.workers.dev"
response = requests.get(worker_url)
token = response.json()['access_token']

# Now use the token for Amadeus API calls
headers = {'Authorization': f'Bearer {token}'}
```

## Need Help?

- Cloudflare Workers Docs: https://developers.cloudflare.com/workers/
- Wrangler Docs: https://developers.cloudflare.com/workers/wrangler/
- Amadeus Docs: https://developers.amadeus.com/
