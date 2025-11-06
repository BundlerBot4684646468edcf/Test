import os, re, json
from datetime import date, datetime, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import html
import time

# Setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
OUTSCRAPER_API_KEY = os.getenv("OUTSCRAPER_API_KEY", "")
USE_LLM = bool(OPENAI_API_KEY)
USE_OUTSCRAPER = bool(OUTSCRAPER_API_KEY)

if USE_LLM:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)

if USE_OUTSCRAPER:
    from outscraper import ApiClient
    outscraper_client = ApiClient(api_key=OUTSCRAPER_API_KEY)

# Page config
st.set_page_config(
    page_title="Hotel Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏨"
)

# Delightful CSS with animations - PROFESSIONAL 2025 DESIGN
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* PROFESSIONAL BACKGROUND - Soft, not distracting */
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8eef3 100%);
        padding: 2rem 3rem !important;
    }

    .block-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 2.5rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Professional Header */
    .glass-header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 16px rgba(37,99,235,0.2);
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .app-logo {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* PROFESSIONAL METRIC CARDS - High Contrast, Clean */
    .metric-card {
        background: #ffffff;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        position: relative;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #2563eb, #1e40af);
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(37,99,235,0.15);
        border-color: #2563eb;
    }

    /* WCAG AA Compliant Text Colors */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
        color: #1e293b;
    }

    .metric-positive {
        color: #10b981;
    }

    .metric-negative {
        color: #ef4444;
    }

    .metric-neutral {
        color: #2563eb;
    }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .metric-trend {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.75rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }

    .trend-up {
        background: rgba(16,185,129,0.1);
        color: #059669;
    }

    .trend-down {
        background: rgba(239,68,68,0.1);
        color: #dc2626;
    }

    /* PROFESSIONAL SECTION HEADERS - Clear Hierarchy */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Category Cards with Gradient */
    .category-card {
        background: white;
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
        animation: fadeInUp 0.6s ease;
    }

    .category-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 24px rgba(102,126,234,0.2);
        border-left-color: #667eea;
    }

    .category-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1e293b;
    }

    .category-score {
        font-size: 2.2rem;
        font-weight: 900;
    }

    /* WCAG Compliant Score Colors - Solid, readable */
    .score-excellent {
        color: #10b981;
    }

    .score-good {
        color: #2563eb;
    }

    .score-average {
        color: #f59e0b;
    }

    .score-poor {
        color: #ef4444;
    }

    /* Professional Progress Bar */
    .progress-bar {
        width: 100%;
        height: 10px;
        background: #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    }

    .progress-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease;
        position: relative;
    }

    .progress-excellent { background: #10b981; }
    .progress-good { background: #2563eb; }
    .progress-average { background: #f59e0b; }
    .progress-poor { background: #ef4444; }

    /* Insight Cards */
    .insight-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border-left: 4px solid #f59e0b;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease;
    }

    .insight-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.12);
    }

    .insight-priority-high { border-left-color: #ef4444; }
    .insight-priority-medium { border-left-color: #f59e0b; }
    .insight-priority-low { border-left-color: #3b82f6; }

    /* Review Cards */
    .review-card {
        background: white;
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        border-left: 4px solid #667eea;
        animation: fadeInUp 0.6s ease;
    }

    .review-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 16px 40px rgba(102,126,234,0.25);
        border-left-width: 6px;
    }

    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .css-1d391kg .sidebar-content {
        color: white;
    }

    /* Buttons */
    /* PROFESSIONAL BUTTONS */
    .stButton>button {
        background: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(37,99,235,0.25);
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }

    .stButton>button:hover {
        background: #1e40af;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.35);
    }

    /* Filter Badge */
    .filter-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #2563eb;
        margin: 0.25rem;
    }

    /* Insight Badges */
    .insight-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-critical {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }

    .badge-important {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }

    .badge-info {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
    }

    /* Better Text Contrast */
    .metric-label {
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #1e293b !important;
        margin-bottom: 0.5rem;
    }

    .category-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #0f172a !important;
    }

    /* Dark readable text everywhere */
    p, div {
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def clean_text(s:str)->str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

def get_score_class(score):
    if score >= 85: return "score-excellent", "progress-excellent"
    elif score >= 70: return "score-good", "progress-good"
    elif score >= 50: return "score-average", "progress-average"
    else: return "score-poor", "progress-poor"

def find_place_id(hotel_query:str)->str|None:
    if not GOOGLE_PLACES_API_KEY: return None
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {"input": hotel_query, "inputtype":"textquery", "fields":"place_id,name,formatted_address", "key": GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    c = r.json().get("candidates", [])
    return c[0]["place_id"] if c else None

def fetch_place_reviews(place_id:str)->pd.DataFrame:
    """Fetch Google Reviews"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id":place_id, "fields":"reviews,name,formatted_address,url", "key":GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    result = r.json().get("result", {})
    reviews = result.get("reviews", []) or []
    rows = []
    for rv in reviews:
        ts = rv.get("time")
        d = datetime.utcfromtimestamp(ts).date() if isinstance(ts,(int,float)) else date.today()
        rows.append({
            "date": d,
            "platform": "Google",
            "language": (rv.get("language") or "en").upper()[:2],
            "rating": rv.get("rating"),
            "review_text": rv.get("text"),
            "author_name": rv.get("author_name", "Anonymous"),
            "author_url": rv.get("author_url", "")
        })
    return pd.DataFrame(rows)

def fetch_outscraper_google_reviews(place_id: str, limit: int = 500) -> pd.DataFrame:
    """Fetch Google Reviews via Outscraper (more than 5 reviews!)"""
    if not USE_OUTSCRAPER:
        return pd.DataFrame()

    try:
        results = outscraper_client.google_maps_reviews(
            [place_id],
            reviews_limit=limit,
            language='de',
            sort='newest'
        )

        rows = []
        for place in results:
            for review in place.get('reviews_data', []):
                review_date = review.get('review_timestamp')
                if review_date:
                    try:
                        d = datetime.fromisoformat(review_date.replace('Z', '+00:00')).date()
                    except:
                        d = date.today()
                else:
                    d = date.today()

                rows.append({
                    "date": d,
                    "platform": "Google",
                    "language": (review.get('review_lang') or "de").upper()[:2],
                    "rating": review.get('review_rating', 5),
                    "review_text": review.get('review_text', ''),
                    "author_name": review.get('author_title', 'Anonymous'),
                    "author_url": review.get('author_link', '')
                })

        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Outscraper Google error: {e}")
        return pd.DataFrame()

def auto_find_booking_url(hotel_name: str, city: str = "") -> str:
    """Automatically find Booking.com URL for a hotel using Google Search"""
    if not USE_OUTSCRAPER:
        return ""

    try:
        search_query = f"{hotel_name} {city} site:booking.com".strip()
        results = outscraper_client.google_search(search_query, limit=5)

        for result in results:
            for item in result.get('organic_results', []):
                url = item.get('link', '')
                if 'booking.com/hotel' in url.lower():
                    return url

        return ""
    except Exception as e:
        st.warning(f"⚠️ Automatische Booking.com Suche fehlgeschlagen: {e}")
        return ""

def auto_find_tripadvisor_url(hotel_name: str, city: str = "") -> str:
    """Automatically find TripAdvisor URL for a hotel using Google Search"""
    if not USE_OUTSCRAPER:
        return ""

    try:
        search_query = f"{hotel_name} {city} site:tripadvisor.com hotel review".strip()
        results = outscraper_client.google_search(search_query, limit=5)

        for result in results:
            for item in result.get('organic_results', []):
                url = item.get('link', '')
                if 'tripadvisor.com' in url.lower() and 'hotel_review' in url.lower():
                    return url

        return ""
    except Exception as e:
        st.warning(f"⚠️ Automatische TripAdvisor Suche fehlgeschlagen: {e}")
        return ""

def fetch_outscraper_booking_reviews(hotel_url: str, limit: int = 500) -> pd.DataFrame:
    """Fetch Booking.com Reviews via Outscraper"""
    if not USE_OUTSCRAPER:
        return pd.DataFrame()

    try:
        # Outscraper Booking.com API
        results = outscraper_client.scrape_site(
            'booking',
            [hotel_url],
            limit=limit
        )

        rows = []
        for result in results:
            for review in result.get('reviews', []):
                review_date_str = review.get('date', '')
                try:
                    d = datetime.strptime(review_date_str, '%Y-%m-%d').date()
                except:
                    d = date.today()

                rows.append({
                    "date": d,
                    "platform": "Booking.com",
                    "language": (review.get('language') or "de").upper()[:2],
                    "rating": float(review.get('rating', 5)) / 2,  # Booking uses 10-point scale
                    "review_text": review.get('positive', '') + ' ' + review.get('negative', ''),
                    "author_name": review.get('author', 'Anonymous'),
                    "author_url": hotel_url
                })

        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Outscraper Booking error: {e}")
        return pd.DataFrame()

def fetch_outscraper_tripadvisor_reviews(hotel_url: str, limit: int = 500) -> pd.DataFrame:
    """Fetch TripAdvisor Reviews via Outscraper"""
    if not USE_OUTSCRAPER:
        return pd.DataFrame()

    try:
        # Outscraper TripAdvisor API
        results = outscraper_client.scrape_site(
            'tripadvisor',
            [hotel_url],
            limit=limit
        )

        rows = []
        for result in results:
            for review in result.get('reviews', []):
                review_date_str = review.get('date', '')
                try:
                    d = datetime.strptime(review_date_str, '%Y-%m-%d').date()
                except:
                    d = date.today()

                rows.append({
                    "date": d,
                    "platform": "TripAdvisor",
                    "language": (review.get('language') or "en").upper()[:2],
                    "rating": review.get('rating', 5),
                    "review_text": review.get('text', ''),
                    "author_name": review.get('author', 'Anonymous'),
                    "author_url": review.get('author_url', hotel_url)
                })

        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"⚠️ Outscraper TripAdvisor error: {e}")
        return pd.DataFrame()

# ✅ REMOVED: No more mock data!
# Only real data from Google Places API, Outscraper, or CSV upload

def llm_analyze_with_insights(df: pd.DataFrame, hotel_name: str, progress_bar=None, status_text=None):
    """Real AI Analysis with OpenAI GPT-4"""
    if not USE_LLM or df.empty:
        return None

    if progress_bar:
        progress_bar.progress(0.2)
    if status_text:
        status_text.text("🤖 OpenAI GPT-4 analysiert Reviews...")

    # Calculate REAL statistics from ALL reviews
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    overall_score = int((avg_rating / 5) * 100)

    # Calculate REAL trend from data (last 30 days vs previous 30 days)
    df_sorted = df.sort_values('date', ascending=False)
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    today = pd.Timestamp.now()
    last_30 = df_sorted[df_sorted['date'] >= (today - pd.Timedelta(days=30))]
    prev_30 = df_sorted[(df_sorted['date'] >= (today - pd.Timedelta(days=60))) &
                        (df_sorted['date'] < (today - pd.Timedelta(days=30)))]

    if len(prev_30) > 0:
        last_30_avg = last_30['rating'].mean() if len(last_30) > 0 else avg_rating
        prev_30_avg = prev_30['rating'].mean()
        trend_pct = ((last_30_avg - prev_30_avg) / prev_30_avg) * 100
        trend = f"{'+' if trend_pct > 0 else ''}{trend_pct:.1f}%"
    else:
        trend = "+0%"

    # Analyze a LARGER sample (50 reviews instead of 20)
    sample_size = min(50, len(df))
    sample = df.head(sample_size)
    reviews_text = []
    for _, r in sample.iterrows():
        reviews_text.append({
            "text": clean_text(r["review_text"]),
            "rating": int(r.get("rating", 5)),
            "author": r.get("author_name", "Anonymous"),
            "date": str(r.get("date", "2024"))
        })

    if progress_bar:
        progress_bar.progress(0.4)

    system = f"""Du bist ein Hotel-Analyse-Experte für {hotel_name}.
Analysiere die Reviews professionell und erstelle einen detaillierten Bericht.

WICHTIG:
- Verwende ECHTE Daten aus den Reviews
- Zitiere ECHTE Reviewtexte als Belege
- Berechne keine eigenen Scores - die werden aus den Daten berechnet"""

    user = f"""
Analysiere diese {sample_size} Hotel-Reviews (von insgesamt {total_reviews} Reviews).

STATISTIKEN (BEREITS BERECHNET):
- Gesamt-Score: {overall_score}/100
- Durchschnittliche Bewertung: {avg_rating:.1f}/5
- Trend: {trend}

Erstelle einen Bericht im EXAKTEN JSON-Format (VERWENDE DIE BERECHNETEN WERTE!):

{{
  "overall_rating": {overall_score},
  "sentiment_score": {overall_score},
  "trend": "{trend}",
  "categories": {{
    "Service": 88,
    "Zimmer": 78,
    "Lage": 92,
    "Gastronomie": 75,
    "Personal": 91,
    "Sauberkeit": 84,
    "Preis-Leistung": 79,
    "Ausstattung": 81,
    "Komfort": 83
  }},
  "insights": [
    {{
      "title": "Konkreter Punkt",
      "text": "Detaillierte Beschreibung mit Zahlen und Fakten",
      "priority": "high|medium|low",
      "evidence": {{
        "mentions": 10,
        "quotes": [
          {{
            "text": "Originaltext",
            "source": "Google Reviews",
            "rating": 7,
            "date": "2024-10",
            "author": "Name"
          }}
        ]
      }}
    }}
  ],
  "recommendations": {{
    "immediate": [{{"action": "Maßnahme", "evidence": "Begründung"}}],
    "short_term": [{{...}}],
    "long_term": [{{...}}]
  }}
}}

REVIEWS:
{json.dumps(reviews_text, ensure_ascii=False)}
"""

    if progress_bar:
        progress_bar.progress(0.6)

    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )

        if progress_bar:
            progress_bar.progress(0.8)

        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("✅ Analyse abgeschlossen!")

        return json.loads(content)
    except Exception as e:
        st.warning(f"LLM Analysis failed: {e}")
        return None

# ============================================
# SIDEBAR - FILTERS
# ============================================

st.sidebar.markdown("""
<div style='text-align: center; padding: 2rem 0; color: white;'>
    <h1 style='font-size: 2rem; font-weight: 900; margin-bottom: 0.5rem;'>🎛️ Filter</h1>
    <p style='opacity: 0.9; font-size: 0.9rem;'>Passe deine Analyse an</p>
</div>
""", unsafe_allow_html=True)

# Date Range Filter
st.sidebar.markdown("### 📅 Zeitraum")
date_range = st.sidebar.selectbox(
    "Wähle Zeitraum",
    ["Alle", "Letzte 30 Tage", "Letzte 90 Tage", "Letzte 6 Monate", "Letztes Jahr", "Benutzerdefiniert"]
)

if date_range == "Benutzerdefiniert":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Von", date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("Bis", date.today())
else:
    end_date = date.today()
    if date_range == "Letzte 30 Tage":
        start_date = end_date - timedelta(days=30)
    elif date_range == "Letzte 90 Tage":
        start_date = end_date - timedelta(days=90)
    elif date_range == "Letzte 6 Monate":
        start_date = end_date - timedelta(days=180)
    elif date_range == "Letztes Jahr":
        start_date = end_date - timedelta(days=365)
    else:
        start_date = None

# Rating Filter
st.sidebar.markdown("### ⭐ Bewertung")
rating_filter = st.sidebar.slider("Nur Reviews mit Rating", 1, 5, (1, 5))

# Language Filter
st.sidebar.markdown("### 🌍 Sprache")
language_filter = st.sidebar.multiselect(
    "Sprachen",
    ["DE", "EN", "IT", "FR", "ES"],
    default=["DE", "EN", "IT"]
)

# Platform Filter
st.sidebar.markdown("### 📱 Plattform")
platform_filter = st.sidebar.multiselect(
    "Datenquellen",
    ["Google", "Booking.com", "TripAdvisor"],
    default=["Google", "Booking.com", "TripAdvisor"]
)

st.sidebar.markdown("---")

# Active Filters Display
if start_date or rating_filter != (1, 5) or len(language_filter) < 5:
    st.sidebar.markdown("### 🎯 Aktive Filter")
    if start_date:
        st.sidebar.markdown(f"""
        <span class="filter-badge">📅 {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}</span>
        """, unsafe_allow_html=True)
    if rating_filter != (1, 5):
        st.sidebar.markdown(f"""
        <span class="filter-badge">⭐ {rating_filter[0]}-{rating_filter[1]} Sterne</span>
        """, unsafe_allow_html=True)
    if len(language_filter) < 5:
        st.sidebar.markdown(f"""
        <span class="filter-badge">🌍 {', '.join(language_filter)}</span>
        """, unsafe_allow_html=True)

# ============================================
# MAIN APP
# ============================================

# Header
st.markdown("""
<div class="glass-header">
    <div class="app-logo">🏨 Hotel Intelligence Dashboard</div>
    <div class="app-subtitle">KI-gestützte Multi-Plattform-Analyse • OpenAI GPT-4 • Echtzeit-Insights</div>
</div>
""", unsafe_allow_html=True)

# Session State
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.results = None

# Input Section
if not st.session_state.analysis_done:
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        hotel_name = st.text_input("🏨 Hotel Name", placeholder="z.B. Hotel Adler")

    with col2:
        hotel_city = st.text_input("📍 Stadt", placeholder="z.B. München")

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 Analysieren", use_container_width=True)

    # ✅ Outscraper Multi-Platform Option (AUTOMATIC)
    reviews_limit = 1000

    if USE_OUTSCRAPER:
        st.markdown("---")
        st.markdown("### 🌐 Automatische Multi-Plattform Analyse")
        st.markdown("✅ **Google Reviews** (via Outscraper)")
        st.markdown("✅ **Booking.com Reviews** (automatisch gesucht)")
        st.markdown("✅ **TripAdvisor Reviews** (automatisch gesucht)")

        reviews_limit = st.slider("Max. Reviews pro Plattform", 100, 5000, 1000, 100)

    # ✅ CSV Upload Option
    st.markdown("---")
    st.markdown("### 📤 Zusätzliche Reviews hochladen")
    st.markdown("Lade CSV-Dateien mit Reviews von Booking.com, TripAdvisor etc. hoch")

    uploaded_files = st.file_uploader(
        "CSV-Dateien (Format: date, platform, language, rating, review_text, author_name)",
        type=["csv"],
        accept_multiple_files=True,
        help="Spalten: date (YYYY-MM-DD), platform, language (DE/EN/IT), rating (1-5), review_text, author_name"
    )

    if analyze_btn and hotel_name and hotel_city:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 Suche Hotel...")

        all_dataframes = []

        # Load Google Reviews (REAL DATA)
        pid = None
        if GOOGLE_PLACES_API_KEY:
            pid = find_place_id(f"{hotel_name} {hotel_city}")
            if pid:
                progress_bar.progress(0.1)
                status_text.text("📥 Lade Reviews von Google Places API...")

                df_google = fetch_place_reviews(pid)
                if not df_google.empty:
                    all_dataframes.append(df_google)
                    st.success(f"✅ {len(df_google)} Google Reviews geladen (Google Places API)")

        # Load Google Reviews via Outscraper (UNLIMITED!)
        if USE_OUTSCRAPER:
            # If no Place ID found via Google Places API, search via Outscraper
            if not pid:
                progress_bar.progress(0.15)
                status_text.text("🔍 Suche Hotel auf Google Maps...")
                try:
                    search_results = outscraper_client.google_maps_search(
                        f"{hotel_name} {hotel_city}",
                        limit=1,
                        language='de'
                    )
                    if search_results and len(search_results) > 0:
                        pid = search_results[0].get('place_id')
                        if pid:
                            st.info(f"🎯 Hotel auf Google Maps gefunden: {search_results[0].get('name', '')}")
                except Exception as e:
                    st.warning(f"⚠️ Google Maps Suche fehlgeschlagen: {e}")

            if pid:
                progress_bar.progress(0.2)
                status_text.text("📥 Lade ALLE Google Reviews via Outscraper...")

                df_outscraper_google = fetch_outscraper_google_reviews(pid, limit=reviews_limit)
                if not df_outscraper_google.empty:
                    all_dataframes.append(df_outscraper_google)
                    st.success(f"✅ {len(df_outscraper_google)} Google Reviews geladen (Outscraper)")
            else:
                st.warning("⚠️ Hotel auf Google Maps nicht gefunden")

        # Load Booking.com Reviews via Outscraper (AUTOMATIC SEARCH)
        if USE_OUTSCRAPER:
            progress_bar.progress(0.4)
            status_text.text("🔍 Suche Hotel auf Booking.com...")

            booking_url = auto_find_booking_url(hotel_name, hotel_city)
            if booking_url:
                st.info(f"🎯 Booking.com gefunden: {booking_url[:80]}...")
                status_text.text("📥 Lade Booking.com Reviews via Outscraper...")

                df_booking = fetch_outscraper_booking_reviews(booking_url, limit=reviews_limit)
                if not df_booking.empty:
                    all_dataframes.append(df_booking)
                    st.success(f"✅ {len(df_booking)} Booking.com Reviews geladen")
            else:
                st.warning("⚠️ Hotel auf Booking.com nicht automatisch gefunden")

        # Load TripAdvisor Reviews via Outscraper (AUTOMATIC SEARCH)
        if USE_OUTSCRAPER:
            progress_bar.progress(0.6)
            status_text.text("🔍 Suche Hotel auf TripAdvisor...")

            tripadvisor_url = auto_find_tripadvisor_url(hotel_name, hotel_city)
            if tripadvisor_url:
                st.info(f"🎯 TripAdvisor gefunden: {tripadvisor_url[:80]}...")
                status_text.text("📥 Lade TripAdvisor Reviews via Outscraper...")

                df_tripadvisor = fetch_outscraper_tripadvisor_reviews(tripadvisor_url, limit=reviews_limit)
                if not df_tripadvisor.empty:
                    all_dataframes.append(df_tripadvisor)
                    st.success(f"✅ {len(df_tripadvisor)} TripAdvisor Reviews geladen")
            else:
                st.warning("⚠️ Hotel auf TripAdvisor nicht automatisch gefunden")

        # Load uploaded CSV files (REAL DATA from user)
        if uploaded_files:
            progress_bar.progress(0.7)
            status_text.text("📥 Lade hochgeladene CSV-Dateien...")

            for uploaded_file in uploaded_files:
                try:
                    df_csv = pd.read_csv(uploaded_file)
                    # Convert date column to datetime
                    df_csv['date'] = pd.to_datetime(df_csv['date']).dt.date
                    # Add author_url if not present
                    if 'author_url' not in df_csv.columns:
                        df_csv['author_url'] = ""
                    all_dataframes.append(df_csv)
                    st.success(f"✅ {len(df_csv)} Reviews aus {uploaded_file.name} geladen")
                except Exception as e:
                    st.error(f"❌ Fehler beim Laden von {uploaded_file.name}: {e}")

        # Combine all reviews (ONLY REAL DATA!)
        if all_dataframes:
            df_reviews = pd.concat(all_dataframes, ignore_index=True)
        else:
            df_reviews = pd.DataFrame()

        if not df_reviews.empty:
            progress_bar.progress(0.8)
            status_text.text("🤖 Starte KI-Analyse mit OpenAI GPT-4...")

            analysis = llm_analyze_with_insights(df_reviews, hotel_name, progress_bar, status_text)

            if analysis:
                st.session_state.results = {
                    "hotel_name": hotel_name,
                    "hotel_city": hotel_city,
                    "analysis": analysis,
                    "reviews": df_reviews
                }
                st.session_state.analysis_done = True

                progress_bar.empty()
                status_text.empty()

                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ KI-Analyse fehlgeschlagen.")
        else:
            st.warning("⚠️ Keine Reviews gefunden. Bitte überprüfe deine API Keys und Eingaben.")

# Results Section
else:
    results = st.session_state.results
    analysis = results["analysis"]
    hotel_name = results["hotel_name"]
    hotel_city = results["hotel_city"]
    df_reviews = results["reviews"]

    # Apply Filters
    df_filtered = df_reviews.copy()

    # Apply Filters with Debug Info
    original_count = len(df_filtered)

    # Date filter
    if start_date:
        df_filtered['date'] = pd.to_datetime(df_filtered['date'])
        df_filtered = df_filtered[
            (df_filtered['date'] >= pd.Timestamp(start_date)) &
            (df_filtered['date'] <= pd.Timestamp(end_date))
        ]
        df_filtered['date'] = df_filtered['date'].dt.date  # Convert back to date

    # Rating filter
    df_filtered = df_filtered[
        (df_filtered['rating'] >= rating_filter[0]) &
        (df_filtered['rating'] <= rating_filter[1])
    ]

    # Language filter
    if language_filter:
        df_filtered = df_filtered[df_filtered['language'].isin(language_filter)]

    # Platform filter
    if platform_filter:
        df_filtered = df_filtered[df_filtered['platform'].isin(platform_filter)]

    # Show filter status
    filtered_count = len(df_filtered)
    if filtered_count < original_count:
        st.info(f"🎯 Filter aktiv: {original_count} → {filtered_count} Reviews ({original_count - filtered_count} herausgefiltert)")

    # Hotel Header
    st.markdown(f"""
    <div style='text-align: center; margin: 2rem 0; animation: fadeInDown 0.6s ease;'>
        <h1 style='font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>{hotel_name}</h1>
        <p style='font-size: 1.3rem; color: #64748b;'>{hotel_city} • {len(df_filtered)} / {len(df_reviews)} Bewertungen (gefiltert)</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    overall_rating = analysis.get('overall_rating', 85)
    sentiment_score = analysis.get('sentiment_score', 82)
    trend = analysis.get('trend', '+3%')

    with col1:
        score_class, _ = get_score_class(overall_rating)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gesamt-Score</div>
            <div class="metric-value {score_class}">{overall_rating}</div>
            <div class="metric-trend trend-up">↑ {trend} vs. Vormonat</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_rating = df_filtered['rating'].mean() if 'rating' in df_filtered and len(df_filtered) > 0 else 4.5
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ø Bewertung</div>
            <div class="metric-value metric-positive">{avg_rating:.1f}</div>
            <div class="metric-trend">von 5.0 Sternen</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        platforms_count = df_filtered['platform'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Plattformen</div>
            <div class="metric-value metric-neutral">{platforms_count}</div>
            <div class="metric-trend">Datenquellen aktiv</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_reviews = len(df_filtered)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Reviews</div>
            <div class="metric-value metric-neutral">{total_reviews}</div>
            <div class="metric-trend">Nach Filter</div>
        </div>
        """, unsafe_allow_html=True)

    # SENTIMENT TIMELINE
    st.markdown('<div class="section-header">📈 Sentiment-Zeitverlauf</div>', unsafe_allow_html=True)

    if len(df_filtered) > 0:
        # Prepare timeline data
        df_timeline = df_filtered.copy()
        df_timeline['date'] = pd.to_datetime(df_timeline['date'])
        df_timeline = df_timeline.sort_values('date')
        df_timeline['sentiment'] = (df_timeline['rating'] - 1) / 4 * 100  # Convert 1-5 to 0-100

        # Group by week
        df_timeline['week'] = df_timeline['date'].dt.to_period('W').apply(lambda r: r.start_time)
        timeline_grouped = df_timeline.groupby('week').agg({
            'sentiment': 'mean',
            'rating': 'count'
        }).reset_index()

        fig_timeline = go.Figure()

        # Add sentiment line
        fig_timeline.add_trace(go.Scatter(
            x=timeline_grouped['week'],
            y=timeline_grouped['sentiment'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea'),
            fill='tozeroy',
            fillcolor='rgba(102,126,234,0.1)'
        ))

        # Add trend line
        z = np.polyfit(range(len(timeline_grouped)), timeline_grouped['sentiment'], 1)
        p = np.poly1d(z)
        fig_timeline.add_trace(go.Scatter(
            x=timeline_grouped['week'],
            y=p(range(len(timeline_grouped))),
            mode='lines',
            name='Trend',
            line=dict(color='#764ba2', width=2, dash='dash')
        ))

        fig_timeline.update_layout(
            title="Sentiment-Entwicklung über Zeit",
            xaxis_title="Datum",
            yaxis_title="Sentiment Score (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            hovermode='x unified',
            font=dict(family='Inter')
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

    # Charts Section
    st.markdown('<div class="section-header">📊 Performance Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        # Platform Distribution
        platform_dist = df_filtered['platform'].value_counts()
        fig_platforms = go.Figure(data=[
            go.Pie(
                labels=platform_dist.index,
                values=platform_dist.values,
                hole=0.4,
                marker=dict(colors=['#667eea', '#f59e0b', '#10b981'])
            )
        ])
        fig_platforms.update_layout(
            title="Reviews nach Plattform",
            height=300,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_platforms, use_container_width=True)

    with chart_col2:
        # Rating Distribution
        rating_dist = df_filtered['rating'].value_counts().sort_index()
        fig_ratings = go.Figure(data=[
            go.Bar(
                x=rating_dist.index,
                y=rating_dist.values,
                marker_color=['#ef4444', '#f59e0b', '#fbbf24', '#3b82f6', '#10b981'],
                text=rating_dist.values,
                textposition='outside'
            )
        ])
        fig_ratings.update_layout(
            title="Rating-Verteilung",
            xaxis_title="Sterne",
            yaxis_title="Anzahl",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    with chart_col3:
        # Language Distribution
        lang_dist = df_filtered['language'].value_counts()
        fig_lang = go.Figure(data=[
            go.Bar(
                x=lang_dist.index,
                y=lang_dist.values,
                marker_color='#667eea',
                text=lang_dist.values,
                textposition='outside'
            )
        ])
        fig_lang.update_layout(
            title="Sprachen",
            xaxis_title="Sprache",
            yaxis_title="Anzahl",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    # Categories (existing code continues...)
    st.markdown('<div class="section-header">🎯 Kategorien-Bewertungen</div>', unsafe_allow_html=True)

    categories = analysis.get("categories", {})
    cols = st.columns(3)
    for idx, (cat_name, score) in enumerate(categories.items()):
        score_class, progress_class = get_score_class(score)
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="category-card">
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                    <div class="category-name">{cat_name}</div>
                    <div class="category-score {score_class}">{score}</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {progress_class}" style="width: {score}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # INSIGHTS SECTION
    st.markdown('<div class="section-header">💡 Kritische Erkenntnisse (KI-generiert)</div>', unsafe_allow_html=True)

    insights = analysis.get("insights", [])
    for insight in insights:
        priority = insight.get("priority", "medium")
        badge_class = {"high": "badge-critical", "medium": "badge-important", "low": "badge-info"}.get(priority, "badge-info")
        priority_text = {"high": "Kritisch", "medium": "Wichtig", "low": "Info"}.get(priority, "Info")
        insight_class = f"insight-priority-{priority}"

        evidence = insight.get("evidence", {})
        quotes = evidence.get("quotes", [])

        with st.container():
            st.markdown(f"""
            <div class="insight-card {insight_class}">
                <div>
                    <span class="insight-badge {badge_class}">{priority_text}</span>
                </div>
                <div class="insight-title" style="font-size: 1.25rem; font-weight: 700; color: #1e293b; margin: 1rem 0 0.5rem 0;">{insight.get('title', '')}</div>
                <div class="insight-text" style="color: #475569; line-height: 1.6;">{insight.get('text', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            if quotes:
                st.markdown(f"**📝 Belege aus {evidence.get('mentions', 0)} Bewertungen (klickbar):**")
                for quote in quotes:
                    author_name = quote.get('author', 'Anonymous')
                    quote_text = quote.get('text', '')

                    matching_review = None
                    if not df_filtered.empty:
                        matches = df_filtered[df_filtered['author_name'] == author_name]
                        if not matches.empty:
                            matching_review = matches.iloc[0]
                        else:
                            for idx, review in df_filtered.iterrows():
                                if quote_text[:50] in review.get('review_text', ''):
                                    matching_review = review
                                    break

                    if matching_review is not None and 'author_url' in matching_review and matching_review['author_url']:
                        review_url = matching_review['author_url']
                    else:
                        review_url = f"https://www.google.com/maps/search/{hotel_name.replace(' ', '+')}+{hotel_city.replace(' ', '+')}"

                    with st.container():
                        st.markdown(f"""
                        <a href="{review_url}" target="_blank" style="text-decoration: none;">
                            <div style='background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 3px solid #667eea; cursor: pointer; transition: all 0.3s ease;' onmouseover="this.style.boxShadow='0 4px 12px rgba(102,126,234,0.3)'; this.style.transform='translateY(-2px)';" onmouseout="this.style.boxShadow=''; this.style.transform='translateY(0)';">
                                <p style='font-style: italic; color: #1e293b; margin-bottom: 0.5rem; font-weight: 500;'>"{quote.get('text', '')}"</p>
                                <p style='font-size: 0.875rem; color: #64748b;'><strong>{author_name}</strong> • {quote.get('source', 'Google')} • {quote.get('date', '2024')} • ⭐ {quote.get('rating', 5)}/10</p>
                                <p style='font-size: 0.75rem; color: #667eea; margin-top: 0.5rem; font-weight: 600;'>🔗 Zum Original-Review →</p>
                            </div>
                        </a>
                        """, unsafe_allow_html=True)

    # RECOMMENDATIONS SECTION
    st.markdown('<div class="section-header">✅ Handlungsempfehlungen (KI-generiert)</div>', unsafe_allow_html=True)

    recommendations = analysis.get("recommendations", {})
    action_categories = [
        ("immediate", "🚨 Sofort umsetzen", "category-immediate"),
        ("short_term", "📅 Kurzfristig (1-3 Monate)", "category-short"),
        ("long_term", "🎯 Langfristig (6+ Monate)", "category-long")
    ]

    for key, title, css_class in action_categories:
        actions = recommendations.get(key, [])
        if actions:
            st.markdown(f"""
            <div class="action-card" style='background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <div class="action-header" style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                    <div class="action-title" style='font-size: 1.25rem; font-weight: 700; color: #1e293b;'>{title}</div>
                    <div class="action-category {css_class}" style='padding: 0.5rem 1rem; border-radius: 20px; background: #667eea; color: white; font-weight: 600; font-size: 0.875rem;'>{len(actions)} Maßnahmen</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for action in actions:
                st.markdown(f"""
                <div style='background: #f8fafc; padding: 1.25rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea;'>
                    <p style='font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; font-size: 1.05rem;'>✓ {action.get('action', '')}</p>
                    <p style='color: #475569; font-size: 0.95rem; line-height: 1.6;'>💡 {action.get('evidence', '')}</p>
                </div>
                """, unsafe_allow_html=True)

    # ALL REVIEWS SECTION
    st.markdown('<div class="section-header">📝 Alle Bewertungen</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color: #475569; margin-bottom: 2rem; font-size: 1.05rem;'>Klicken Sie auf eine Bewertung um zur Original-Quelle zu gelangen • <strong>{len(df_filtered)}</strong> Reviews gefiltert</p>", unsafe_allow_html=True)

    for idx, review in df_filtered.head(20).iterrows():
        rating = review.get('rating', 5)
        stars = "⭐" * int(rating)

        author_url = review.get("author_url", "")
        if not author_url:
            author_url = f"https://www.google.com/maps/search/{hotel_name.replace(' ', '+')}+{hotel_city.replace(' ', '+')}"

        platform = review.get('platform', 'Google')
        platform_emoji = {"Google": "🔵", "Booking.com": "🏨", "TripAdvisor": "✈️"}.get(platform, "📱")

        # Safe text handling
        review_text = review.get('review_text') or ''
        review_text = str(review_text)
        text_preview = review_text[:300]
        show_more = '...' if len(review_text) > 300 else ''

        st.markdown(f"""
        <a href="{author_url}" target="_blank" style="text-decoration: none;">
            <div style='background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #2563eb; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.06);' onmouseover="this.style.boxShadow='0 8px 20px rgba(37,99,235,0.25)'; this.style.transform='translateY(-3px)';" onmouseout="this.style.boxShadow='0 2px 8px rgba(0,0,0,0.06)'; this.style.transform='translateY(0)';">
                <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;'>
                    <div>
                        <p style='font-weight: 700; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.25rem;'>{review.get('author_name', 'Anonymous')}</p>
                        <p style='font-size: 0.875rem; color: #64748b;'>{platform_emoji} {platform} • {review.get('language', 'DE')} • {review.get('date', '2024')}</p>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 1.5rem;'>{stars}</div>
                        <div style='font-weight: 700; color: #2563eb; font-size: 1.1rem;'>{rating}/5</div>
                    </div>
                </div>
                <p style='color: #334155; line-height: 1.7; font-size: 0.95rem;'>{text_preview}{show_more}</p>
                <p style='font-size: 0.85rem; color: #2563eb; margin-top: 1rem; font-weight: 600;'>🔗 Vollständiges Review lesen →</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

    # Reset button
    st.write("")
    st.write("")
    if st.button("🔄 Neues Hotel analysieren"):
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()
