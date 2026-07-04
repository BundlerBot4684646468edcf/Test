# Quick Start: Salon Booking SaaS

## 1. Set up environment

```bash
# Clone or navigate to the repo
cd /home/user/Test

# Install dependencies
pip install -r requirements.txt
```

## 2. Check your Cal.com connection

Before doing anything, verify your API key and connection:

```bash
export CAL_API_KEY="cal_YOUR_KEY_HERE"
export CAL_API_BASE="https://api.cal.com/v2"  # default, adjust if needed

# Run the debug tester
python debug_booking.py
```

This will test:
- ✅ API key is valid
- ✅ Connection to Cal.com
- ✅ Team creation
- ✅ Member invite
- ✅ Event type creation
- ✅ Slot retrieval
- ✅ Booking creation

**If any step fails**, the error message from Cal.com will tell you exactly what's wrong (auth, payload format, etc.).

## 3. Run the backend server

```bash
# Set environment variables
export CAL_API_KEY="cal_YOUR_KEY_HERE"
export SALON_DATABASE_URL="sqlite:///salon.db"  # or your Postgres URL

# Start the FastAPI server
uvicorn salon.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be at `http://localhost:8000`

**Quick connectivity test:**
```bash
curl http://localhost:8000/debug/cal-ping
```

Should return: `{"ok": true, "cal_user": {...}}`

## 4. Test via HTTP (manual)

Onboard a salon:
```bash
curl -X POST http://localhost:8000/salons \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Salon",
    "slug": "test-salon",
    "timezone": "Europe/Berlin",
    "employees": [
      {"name": "Alice", "email": "alice@salon.test"}
    ],
    "services": [
      {"name": "Haircut", "duration_min": 45}
    ]
  }'
```

Get availability:
```bash
curl -X POST http://localhost:8000/famulor/tools/get_availability \
  -H "Content-Type: application/json" \
  -d '{
    "salon_slug": "test-salon",
    "service_name": "Haircut",
    "date_from": "2026-07-01",
    "date_to": "2026-07-07"
  }'
```

Book an appointment:
```bash
curl -X POST http://localhost:8000/famulor/tools/book_appointment \
  -H "Content-Type: application/json" \
  -d '{
    "salon_slug": "test-salon",
    "service_name": "Haircut",
    "start_at": "2026-07-01T09:00:00",
    "customer_name": "John Doe",
    "customer_email": "john@example.com"
  }'
```

## 5. Run tests

```bash
python -m pytest tests/ -v
```

Should see: **10 passed**

## Troubleshooting

| Issue | Check |
|---|---|
| `401 Unauthorized` from Cal.com | API key is wrong or expired |
| `422 Unprocessable Entity` | Payload format doesn't match Cal.com v2 API |
| `404 Not Found` for salon/service/employee | Check spelling in the request (case-insensitive) |
| Server crashes on `/salons` | Run `python debug_booking.py` first to isolate the Cal.com issue |
| Famulor not calling the tools | Check Famulor's tool configuration (point to `/famulor/tools/get_availability` and `/famulor/tools/book_appointment`) |

## Next Steps

1. **Verify Cal.com works** → run `python debug_booking.py`
2. **Start the server** → `uvicorn salon.main:app --reload`
3. **Onboard your first real salon** → POST `/salons`
4. **Connect Famulor** → configure function-calling to call `/famulor/tools/*`
5. **Set up Cal.com Workflows** → see `docs/process_flow.md` for SMS reminders + review requests

## Documentation

- `CLAUDE.md` — Full project architecture & context
- `docs/process_flow.md` — End-to-end flow from call to SMS
