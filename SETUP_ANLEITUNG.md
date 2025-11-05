# 🏨 Hotel Intelligence Dashboard - Setup Anleitung

## 📦 Schritt 1: Dateien auf deinen PC kopieren

Du brauchst diese **3 Dateien**:
- ✅ `app_ultimate.py` (Das Dashboard)
- ✅ `.env` (Deine API Keys - BEREITS KONFIGURIERT!)
- ✅ `requirements.txt` (Python Pakete)

**Alle Dateien sind in diesem Ordner:**
```
/home/user/Test/
```

---

## 💻 Schritt 2: Auf deinem LOKALEN PC (Windows/Mac/Linux)

### A) Erstelle einen Ordner
```bash
mkdir hotel-dashboard
cd hotel-dashboard
```

### B) Kopiere die 3 Dateien in diesen Ordner
- `app_ultimate.py`
- `.env` (WICHTIG: Enthält deine API Keys!)
- `requirements.txt`

### C) Installiere Python Pakete
```bash
pip install -r requirements.txt
```

**ODER einzeln:**
```bash
pip install streamlit plotly python-dotenv pandas numpy requests openai outscraper
```

### D) Starte das Dashboard
```bash
python -m streamlit run app_ultimate.py
```

**Oder einfach:**
```bash
streamlit run app_ultimate.py
```

---

## 🎯 Schritt 3: Dashboard öffnet sich automatisch!

Nach dem Start öffnet sich automatisch dein Browser auf:
```
http://localhost:8501
```

Falls nicht, öffne den Link manuell im Browser.

---

## 🔑 Wichtig: Deine API Keys sind BEREITS konfiguriert!

Die `.env` Datei enthält bereits:
- ✅ **OUTSCRAPER_API_KEY** - Dein Outscraper Key ist drin!
- ⚠️ **OPENAI_API_KEY** - Du musst noch deinen OpenAI Key hinzufügen
- ⚠️ **GOOGLE_PLACES_API_KEY** - Optional (für Google Reviews)

### So fügst du deinen OpenAI Key hinzu:

Öffne die `.env` Datei mit Notepad/TextEdit:
```
OPENAI_API_KEY=dein_openai_key_hier
```

Hol dir einen OpenAI Key: https://platform.openai.com/api-keys

---

## 🚀 Schritt 4: Hotel analysieren!

Im Dashboard:

1. **Hotel Name eingeben:** z.B. "Hotel Adler"
2. **Stadt eingeben:** z.B. "München"
3. **(Optional) Booking.com URL:** z.B. `https://www.booking.com/hotel/de/adler-munich.html`
4. **(Optional) TripAdvisor URL:** z.B. `https://www.tripadvisor.com/Hotel_Review-...`
5. **Klick "🚀 Analysieren"**

Das System wird:
- ✅ Google Reviews holen via Outscraper (UNLIMITED!)
- ✅ Booking.com Reviews holen (falls URL angegeben)
- ✅ TripAdvisor Reviews holen (falls URL angegeben)
- 🤖 OpenAI GPT-4 analysiert alles
- 📊 Zeigt Dashboard mit Charts, Insights, Empfehlungen

---

## ❓ Probleme?

### "streamlit command not found"
```bash
python -m streamlit run app_ultimate.py
```

### "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'outscraper'"
```bash
pip install outscraper
```

### Dashboard lädt nicht
- Überprüfe ob `.env` Datei im gleichen Ordner wie `app_ultimate.py` ist
- Überprüfe ob OPENAI_API_KEY in `.env` gesetzt ist

---

## 💰 API Kosten

| Service | Kosten | Was bekommst du |
|---------|--------|-----------------|
| **Outscraper** | ~$0.002 pro Review | Unbegrenzte Reviews von Google/Booking/TripAdvisor |
| **OpenAI GPT-4** | ~$0.00015 pro 1K tokens | AI Sentiment-Analyse |

**Beispiel:** 500 Reviews von 3 Plattformen (1500 total) = ca. **$3-5**

**GRATIS:** Neue Outscraper Accounts bekommen Credits zum Testen!

---

## ✨ Features

- 📊 Multi-Platform Reviews (Google, Booking, TripAdvisor)
- 🤖 Echte AI-Analyse mit OpenAI GPT-4
- 📈 Sentiment Timeline mit Trends
- 🎛️ Filter: Zeitraum, Rating, Sprache, Plattform
- 💬 Kritische Erkenntnisse mit Beweis-Zitaten
- ✅ Handlungsempfehlungen
- 🎨 Delightful Design mit Glassmorphism

---

**Viel Erfolg! 🚀**
