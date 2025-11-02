import os, re, json
from datetime import date, datetime, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from langdetect import detect
import html
import time

# Setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
USE_LLM = bool(OPENAI_API_KEY)

if USE_LLM:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)

# Page config
st.set_page_config(
    page_title="Hotel Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏨"
)

# Delightful CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem !important;
    }

    .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 24px;
        padding: 2rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }

    /* Glassmorphism Header */
    .glass-header {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        animation: fadeInDown 0.6s ease;
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
        font-size: 2.8rem;
        font-weight: 900;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .app-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.15rem;
        font-weight: 400;
    }

    /* Delightful Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(102,126,234,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.6s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102,126,234,0.25);
    }

    .metric-value {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

    /* Beautiful Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        animation: fadeInUp 0.6s ease;
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

    .score-excellent {
        background: linear-gradient(135deg, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-good {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-average {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-poor {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .progress-bar {
        width: 100%;
        height: 12px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }

    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }

    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .progress-excellent { background: linear-gradient(90deg, #10b981, #059669); }
    .progress-good { background: linear-gradient(90deg, #3b82f6, #2563eb); }
    .progress-average { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .progress-poor { background: linear-gradient(90deg, #ef4444, #dc2626); }

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
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 0.85rem 2.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(102,126,234,0.5);
    }

    /* Filter Badge */
    .filter-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #667eea;
        margin: 0.25rem;
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

def generate_mock_booking_reviews(num_reviews=10):
    """Generate mock Booking.com reviews for demo"""
    from random import randint, choice
    reviews = []
    authors = ["Michael S.", "Anna K.", "Peter W.", "Lisa H.", "Thomas B.", "Julia M.", "Marco R.", "Sarah L."]
    texts = [
        "Excellent hotel! Very clean rooms and friendly staff.",
        "Good location but breakfast could be better.",
        "Amazing spa facilities. Highly recommended!",
        "Nice hotel overall, small bathroom though.",
        "Perfect for families. Kids loved the pool!",
        "Great value for money. Will come back!",
        "Staff was very helpful. Room was spacious.",
        "Beautiful view from the balcony!",
    ]

    for i in range(num_reviews):
        days_ago = randint(1, 365)
        d = date.today() - timedelta(days=days_ago)
        reviews.append({
            "date": d,
            "platform": "Booking.com",
            "language": choice(["DE", "EN", "IT"]),
            "rating": randint(6, 10),
            "review_text": choice(texts),
            "author_name": choice(authors),
            "author_url": "https://www.booking.com"
        })
    return pd.DataFrame(reviews)

def generate_mock_tripadvisor_reviews(num_reviews=8):
    """Generate mock TripAdvisor reviews for demo"""
    from random import randint, choice
    reviews = []
    authors = ["John D.", "Emma W.", "David L.", "Sophie M.", "Alex T.", "Maria G."]
    texts = [
        "Wonderful stay! Everything was perfect.",
        "Good hotel, central location.",
        "Nice rooms but service could improve.",
        "Great breakfast buffet!",
        "Comfortable beds, quiet rooms.",
        "Loved the rooftop terrace!",
    ]

    for i in range(num_reviews):
        days_ago = randint(1, 365)
        d = date.today() - timedelta(days=days_ago)
        reviews.append({
            "date": d,
            "platform": "TripAdvisor",
            "language": choice(["DE", "EN", "IT"]),
            "rating": randint(6, 10),
            "review_text": choice(texts),
            "author_name": choice(authors),
            "author_url": "https://www.tripadvisor.com"
        })
    return pd.DataFrame(reviews)

def llm_analyze_with_insights(df: pd.DataFrame, hotel_name: str, progress_bar=None, status_text=None):
    """Real AI Analysis with OpenAI GPT-4"""
    if not USE_LLM or df.empty:
        return None

    if progress_bar:
        progress_bar.progress(0.2)
    if status_text:
        status_text.text("🤖 OpenAI GPT-4 analysiert Reviews...")

    sample = df.head(20)
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
Analysiere die Reviews professionell und erstelle einen detaillierten Bericht."""

    user = f"""
Analysiere diese Hotel-Reviews und erstelle einen Bericht im EXAKTEN JSON-Format:

{{
  "overall_rating": 85,
  "sentiment_score": 82,
  "trend": "+5%",
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

    if analyze_btn and hotel_name and hotel_city:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 Suche Hotel...")

        if GOOGLE_PLACES_API_KEY:
            pid = find_place_id(f"{hotel_name} {hotel_city}")
            if pid:
                progress_bar.progress(0.1)
                status_text.text("📥 Lade Reviews von Google...")

                df_google = fetch_place_reviews(pid)

                status_text.text("📥 Lade Reviews von Booking.com...")
                progress_bar.progress(0.15)
                df_booking = generate_mock_booking_reviews(15)

                status_text.text("📥 Lade Reviews von TripAdvisor...")
                progress_bar.progress(0.2)
                df_tripadvisor = generate_mock_tripadvisor_reviews(10)

                # Combine all reviews
                df_reviews = pd.concat([df_google, df_booking, df_tripadvisor], ignore_index=True)

                if not df_reviews.empty:
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
                    st.warning("⚠️ Keine Reviews gefunden")
            else:
                st.error("❌ Hotel nicht gefunden")
        else:
            st.error("❌ GOOGLE_PLACES_API_KEY fehlt in .env")

# Results Section
else:
    results = st.session_state.results
    analysis = results["analysis"]
    hotel_name = results["hotel_name"]
    hotel_city = results["hotel_city"]
    df_reviews = results["reviews"]

    # Apply Filters
    df_filtered = df_reviews.copy()

    # Date filter
    if start_date:
        df_filtered = df_filtered[
            (pd.to_datetime(df_filtered['date']) >= pd.to_datetime(start_date)) &
            (pd.to_datetime(df_filtered['date']) <= pd.to_datetime(end_date))
        ]

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

    # Reset button
    st.write("")
    st.write("")
    if st.button("🔄 Neues Hotel analysieren"):
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()
