# 🚀 Mundpost Quick Start

## ⏱️ 5 Minuten Setup

### 1. API-Keys besorgen (siehe SETUP.md für Details)

**Google Places API** (https://console.cloud.google.com/)
- [ ] Create Project
- [ ] Enable Places API
- [ ] Create API Key
- [ ] Copy: `AIzaSy...`

**Twilio SMS** (https://console.twilio.com/)
- [ ] Sign Up
- [ ] Buy Phone Number
- [ ] Copy: Account SID, Auth Token, Phone Number

**Resend Email** (https://resend.com/)
- [ ] Sign Up
- [ ] Create API Key
- [ ] Copy: `re_...`

### 2. Keys in `.env` einfügen

```bash
# Öffne .env und fülle aus:
GOOGLE_PLACES_API_KEY=AIzaSy...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
RESEND_API_KEY=re_...
```

### 3. Datenbank starten

```bash
docker-compose -f docker-compose.dev.yml up -d
npm run prisma:migrate
```

### 4. Server starten (2 Terminals)

**Terminal 1 — Backend:**
```bash
npm run dev
# Port: 3000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# Port: 3001
```

### 5. Validieren

```bash
curl http://localhost:3000/api/setup/status
```

Sollte zeigen:
```json
{
  "allConfigured": true,
  "services": {
    "googlePlaces": { "configured": true, "status": "✅ Ready" },
    "sms": { "configured": true, "status": "✅ Ready" },
    "email": { "configured": true, "status": "✅ Ready" }
  }
}
```

---

## 📝 Beispiel: Business + Kunden + SMS versenden

### Business erstellen
```bash
BUSINESS_ID=$(curl -s -X POST http://localhost:3000/api/businesses \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Pizza",
    "ownerName": "Marco",
    "timezone": "Europe/Rome"
  }' | jq -r '.id')

echo "Business ID: $BUSINESS_ID"
```

### Google Place finden
```bash
curl -X POST http://localhost:3000/api/businesses/$BUSINESS_ID/find-place \
  -H "Content-Type: application/json" \
  -d '{
    "businessName": "My Pizza",
    "address": "Via Roma 1, Bolzano, Italy"
  }' | jq .
```

### Foto hochladen (optional)
```bash
# Mit echtem Foto ersetzen
curl -X POST http://localhost:3000/api/businesses/$BUSINESS_ID/photo \
  -F "photo=@/path/to/owner.jpg"
```

### Kunden importieren
```bash
# customers.csv erstellen:
cat > customers.csv << EOF
firstName,phone,email,servedAt
Marco,+393891234567,marco@example.com,2024-01-15
Anna,,anna@example.com,2024-01-14
EOF

curl -X POST http://localhost:3000/api/businesses/$BUSINESS_ID/customers/import \
  -F "file=@customers.csv" \
  -F "source=past"
```

### Kunden-ID abrufen
```bash
curl http://localhost:3000/api/businesses/$BUSINESS_ID/customers | jq '.customers[0]'
# Merke dir die "id"
```

### SMS/Email versenden
```bash
CUSTOMER_ID="xxxxxx"

# SMS:
curl -X POST http://localhost:3000/api/businesses/$BUSINESS_ID/review-requests \
  -H "Content-Type: application/json" \
  -d "{
    \"customerId\": \"$CUSTOMER_ID\",
    \"channel\": \"sms\"
  }"

# Email:
curl -X POST http://localhost:3000/api/businesses/$BUSINESS_ID/review-requests \
  -H "Content-Type: application/json" \
  -d "{
    \"customerId\": \"$CUSTOMER_ID\",
    \"channel\": \"email\"
  }"
```

**Hinweis:** Echte SMS/Email werden sofort gesendet. Logs zeigen Status.

---

## 🌐 Dashboard

Öffne http://localhost:3001
- Login mit deiner Business ID
- Siehe: Kunden, Review Requests, Stats

---

## 🎯 Was funktioniert

✅ SMS mit Foto (MMS) + personalisiertem Text  
✅ Email mit inline Foto + HTML  
✅ Automatische Versand-Cron-Jobs  
✅ Erinnerungen nach 3 Tagen  
✅ GDPR Opt-out  
✅ Wochenbericht mit Chart  
✅ Rate Limiting (daily batch limit)  
✅ CSV Import  

---

## ❓ Fehlersuche

**SMS nicht gesendet?**
- Logs prüfen: `✅ SMS sent` oder `❌ SMS error`
- Twilio Balance check: https://console.twilio.com/

**Email nicht gesendet?**
- Logs prüfen: `✅ Email sent` oder `❌ Email error`
- Prüfe Spam-Folder

**Place nicht gefunden?**
- Google Places API aktiviert? https://console.cloud.google.com/
- Ist der API Key richtig?
- Probiere exakte Adresse

---

## 📚 Mehr

- **Full Setup Guide:** `SETUP.md`
- **API Docs:** Curl-Beispiele in `SETUP.md`
- **Database:** `prisma/schema.prisma`
- **Cron Jobs:** `src/services/cronJobs.ts`

🎉 **Done!**
