# 🚀 Hotel Decision Simulator - SCHNELLSTART

## 📍 Dein Projekt ist hier:

```
/home/user/neue app 123
```

Auf Windows könnte es z.B. hier sein:
```
C:\Users\DeinName\neue app 123
```

---

## ✅ So startest du die App in 5 Schritten:

### Schritt 1: Zum Ordner gehen

**Windows (PowerShell):**
```powershell
cd "C:\Users\DeinName\neue app 123"
```

**Mac/Linux:**
```bash
cd "/home/user/neue app 123"
```

---

### Schritt 2: Pakete installieren

```bash
npm install
```

⏱️ Dauert 1-2 Minuten

---

### Schritt 3: .env Datei erstellen

**Windows:**
```powershell
copy .env.example .env
notepad .env
```

**Mac/Linux:**
```bash
cp .env.example .env
nano .env
```

**Füge das ein:**
```env
# Datenbank (z.B. von neon.tech)
DATABASE_URL="postgresql://user:pass@host/db"

# App-URL
NEXTAUTH_URL="http://localhost:3000"

# Geheimer Schlüssel (irgendein langer Text)
NEXTAUTH_SECRET="mein-super-geheimer-schluessel-12345678901234567890"

# Optional: Für Chatbot
OPENAI_API_KEY="sk-dein-key"
```

---

### Schritt 4: Datenbank vorbereiten

```bash
npx prisma generate
npx prisma db push
```

✅ Du solltest sehen: "Database schema pushed to database"

---

### Schritt 5: Starten! 🎉

```bash
npm run dev
```

Öffne deinen Browser:
**http://localhost:3000**

---

## 🗄️ Datenbank erstellen (einfachste Methode):

### Neon (kostenlos, 2 Minuten):

1. Gehe zu: **https://neon.tech**
2. Registriere dich (kostenlos)
3. Erstelle neues Projekt: "hotel-simulator"
4. Kopiere den Connection String
5. Füge ihn in die `.env` Datei ein

Der Connection String sieht so aus:
```
postgresql://user:password@ep-xxx-xxx.neon.tech/neondb
```

---

## ✅ Testen ob es funktioniert:

1. Öffne: http://localhost:3000
2. Klicke **"Sign up free"**
3. Erstelle einen Account
4. Du siehst das Dashboard! ✅

---

## 📁 Was ist in diesem Ordner?

```
neue app 123/
├── app/                  ← Alle Webseiten
│   ├── page.tsx         ← Startseite
│   ├── (app)/
│   │   ├── dashboard/   ← Dashboard
│   │   ├── simulator/   ← ⭐ SIMULATOR (Hauptfunktion)
│   │   ├── scenarios/   ← Gespeicherte Szenarien
│   │   ├── chat/        ← AI Chat
│   │   └── settings/    ← Einstellungen
│   ├── auth/            ← Login/Register
│   └── api/             ← Backend
├── components/          ← UI-Komponenten
├── lib/                 ← Logik & Funktionen
├── prisma/              ← Datenbank
├── package.json         ← Paket-Liste
└── .env.example         ← Konfigurations-Vorlage
```

---

## ❌ Probleme?

### "npm not found"
→ Node.js installieren: https://nodejs.org

### "Cannot connect to database"
→ Prüfe deine `DATABASE_URL` in der `.env` Datei

### Port 3000 belegt?
```bash
PORT=3001 npm run dev
```

### Chatbot funktioniert nicht?
→ Das ist okay! Alle anderen Funktionen funktionieren ohne OpenAI API Key

---

## 📞 Hilfe?

Schau in die **README.md** für mehr Details.

---

**Viel Erfolg!** 🎉
