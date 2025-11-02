import os, re, json, math
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----------------- Setup -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
AMADEUS_WORKER_URL = os.getenv("AMADEUS_WORKER_URL", "")
USE_LLM = bool(OPENAI_API_KEY)

if USE_LLM:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)

# Page config
st.set_page_config(page_title="🏨 Hotel Intelligence Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS matching the HTML design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e8f0fe 100%) !important; }

    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8f0fe 100%) !important; }

    div[data-testid="stHeader"] { background: transparent; }

    .dashboard-header {
        text-align: center;
        background: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .logo {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0066ff, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border-top: 4px solid #0066ff;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0066ff;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
    }

    .category-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0066ff;
        margin-bottom: 1rem;
    }

    .category-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .category-name {
        font-weight: 700;
        color: #333;
    }

    .category-score {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0066ff;
    }

    .category-bar {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }

    .category-bar-fill {
        height: 100%;
        background: linear-gradient(135deg, #0066ff, #00ccff);
        border-radius: 10px;
    }

    .insight-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .insight-title {
        font-weight: 700;
        color: #333;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    .insight-text {
        color: #666;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .evidence-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .evidence-header {
        font-weight: 600;
        color: #0066ff;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }

    .evidence-quote {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #0066ff;
    }

    .quote-text {
        font-style: italic;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .quote-meta {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }

    .review-link {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.5rem 1rem;
        background: #0066ff;
        color: white !important;
        text-decoration: none;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    .review-link:hover {
        background: #0052cc;
        text-decoration: none;
    }

    .action-group {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .action-group-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .action-immediate { color: #ef4444; }
    .action-short { color: #f59e0b; }
    .action-long { color: #0066ff; }

    .action-item {
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }

    .action-text {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .action-evidence {
        color: #666;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .stButton>button {
        width: 100%;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        background: linear-gradient(135deg, #0066ff, #00ccff);
        color: white;
        border: none;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,102,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="dashboard-header">
    <div class="logo">📊 Hotel Intelligence</div>
    <p style="color: #666; font-size: 1.1rem;">KI-gestützte Hotel-Analyse</p>
</div>
""", unsafe_allow_html=True)

# Helper functions (keeping existing ones)
def clean_text(s:str)->str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

def detect_lang(text:str)->str:
    try:
        l = detect(text or "")
    except Exception:
        l = "en"
    l = l.upper()[:2]
    return l if l in {"DE","IT","EN"} else "EN"

def heuristic_sentiment(text:str)->int:
    POS = {"great","amazing","love","excellent","freundlich","gentile","perfetto","clean","modern","spacious"}
    NEG = {"broken","dirty","cold","slow","unfriendly","loud","schmutzig","pessimo","old","outdated","noisy"}
    t = (text or "").lower()
    pos = any(w in t for w in POS)
    neg = any(w in t for w in NEG)
    return 1 if (pos and not neg) else (-1 if (neg and not pos) else 0)

def fetch_amadeus_token()->str:
    if not AMADEUS_WORKER_URL:
        return ""
    try:
        r = requests.get(AMADEUS_WORKER_URL, timeout=10)
        r.raise_for_status()
        return r.json().get("access_token", "")
    except Exception:
        return ""

def find_place_id(hotel_query:str)->str|None:
    if not GOOGLE_PLACES_API_KEY: return None
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {"input": hotel_query, "inputtype":"textquery", "fields":"place_id,name,formatted_address", "key": GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    c = r.json().get("candidates", [])
    return c[0]["place_id"] if c else None

def fetch_place_reviews(place_id:str)->pd.DataFrame:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id":place_id, "fields":"reviews,name,formatted_address,url", "key":GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    result = r.json().get("result", {})
    reviews = result.get("reviews", []) or []
    rows = []
    for rv in reviews:
        ts = rv.get("time")
        d = datetime.utcfromtimestamp(ts).date().isoformat() if isinstance(ts,(int,float)) else None
        rows.append({
            "date": d,
            "platform": "Google",
            "language": (rv.get("language") or "").upper()[:2] if rv.get("language") else "",
            "rating": rv.get("rating"),
            "review_text": rv.get("text"),
            "author_name": rv.get("author_name", "Anonymous")
        })
    return pd.DataFrame(rows)

def llm_analyze_with_insights(df: pd.DataFrame, hotel_name: str):
    """Generate comprehensive insights with evidence quotes"""
    if not USE_LLM or df.empty:
        return None

    # Take sample of reviews for analysis
    sample = df.head(20)
    reviews_text = []
    for _, r in sample.iterrows():
        reviews_text.append({
            "text": clean_text(r["review_text"]),
            "rating": int(r.get("rating", 5)),
            "author": r.get("author_name", "Anonymous"),
            "date": str(r.get("date", "2024"))
        })

    system = f"""Du bist ein Hotel-Analyse-Experte für {hotel_name}.
Analysiere die Reviews und identifiziere kritische Erkenntnisse mit konkreten Zitaten als Beweis.
Antworte NUR mit validem JSON."""

    user = f"""
Analysiere diese Hotel-Reviews und erstelle einen Bericht im EXAKTEN JSON-Format:

{{
  "overall_rating": 85,
  "sentiment_score": 82,
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
      "title": "Konkreter Punkt (z.B. Check-in Wartezeiten)",
      "text": "Detaillierte Beschreibung mit Zahlen",
      "evidence": {{
        "mentions": 5,
        "quotes": [
          {{
            "text": "Originaltext aus Review",
            "source": "Google Reviews",
            "rating": 7,
            "date": "2024-10",
            "author": "Max M."
          }}
        ]
      }}
    }}
  ],
  "recommendations": {{
    "immediate": [
      {{
        "action": "Konkrete Maßnahme mit Budget und Zeitrahmen",
        "evidence": "Begründung basierend auf Reviews"
      }}
    ],
    "short_term": [
      {{
        "action": "Mittelfristige Maßnahme",
        "evidence": "Begründung"
      }}
    ],
    "long_term": [
      {{
        "action": "Langfristige Investition",
        "evidence": "Begründung mit ROI"
      }}
    ]
  }}
}}

REVIEWS:
{json.dumps(reviews_text, ensure_ascii=False)}

Erstelle 3-5 Insights mit jeweils 1-3 Zitaten als Beweis. Nutze echte Texte aus den Reviews!
"""

    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        content = resp.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        st.warning(f"LLM Analysis failed: {e}")
        return None

# Main App
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.results = None

# Input Section
if not st.session_state.analysis_done:
    col1, col2 = st.columns([2, 1])

    with col1:
        hotel_name = st.text_input("🏨 Hotel Name", placeholder="z.B. Hotel Rosengarten")
        hotel_city = st.text_input("📍 Stadt", placeholder="z.B. Bolzano")

    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analysieren", use_container_width=True)

    if analyze_btn and hotel_name and hotel_city:
        with st.spinner("🔄 Analysiere Hotel-Daten..."):
            # Fetch reviews
            if GOOGLE_PLACES_API_KEY:
                pid = find_place_id(f"{hotel_name} {hotel_city}")
                if pid:
                    df_reviews = fetch_place_reviews(pid)

                    if not df_reviews.empty:
                        # Run LLM analysis
                        analysis = llm_analyze_with_insights(df_reviews, hotel_name)

                        if analysis:
                            st.session_state.results = {
                                "hotel_name": hotel_name,
                                "hotel_city": hotel_city,
                                "analysis": analysis,
                                "reviews": df_reviews
                            }
                            st.session_state.analysis_done = True
                            st.rerun()
                        else:
                            st.error("❌ Analyse fehlgeschlagen. Bitte später erneut versuchen.")
                    else:
                        st.warning("⚠️ Keine Reviews gefunden. Bitte anderen Namen versuchen.")
                else:
                    st.error("❌ Hotel nicht gefunden. Bitte genaueren Namen angeben.")
            else:
                st.error("❌ GOOGLE_PLACES_API_KEY fehlt in .env")

# Results Section
else:
    results = st.session_state.results
    analysis = results["analysis"]
    hotel_name = results["hotel_name"]
    hotel_city = results["hotel_city"]
    df_reviews = results["reviews"]

    # Hotel Name Header
    st.markdown(f"<h1 style='text-align: center; color: #0066ff; margin: 2rem 0;'>{hotel_name}, {hotel_city}</h1>", unsafe_allow_html=True)

    # Stats Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{analysis.get('overall_rating', 85)}</div>
            <div class="stat-label">Gesamt-Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df_reviews)}</div>
            <div class="stat-label">Bewertungen</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{analysis.get('sentiment_score', 82)}%</div>
            <div class="stat-label">Zufriedenheit</div>
        </div>
        """, unsafe_allow_html=True)

    # Categories
    st.markdown("<h2 style='margin-top: 3rem; margin-bottom: 1.5rem;'>📊 Kategorien-Bewertungen</h2>", unsafe_allow_html=True)

    categories = analysis.get("categories", {})
    cols = st.columns(3)
    for idx, (cat_name, score) in enumerate(categories.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="category-card">
                <div class="category-header">
                    <div class="category-name">{cat_name}</div>
                    <div class="category-score">{score}</div>
                </div>
                <div class="category-bar">
                    <div class="category-bar-fill" style="width: {score}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Insights with Evidence
    st.markdown("<h2 style='margin-top: 3rem; margin-bottom: 1.5rem;'>🔍 Kritische Erkenntnisse</h2>", unsafe_allow_html=True)

    insights = analysis.get("insights", [])
    for insight in insights:
        evidence = insight.get("evidence", {})
        quotes = evidence.get("quotes", [])

        quotes_html = ""
        for quote in quotes:
            hotel_slug = hotel_name.lower().replace(" ", "-")
            review_url = f"https://google.com/maps/place/{hotel_slug}"

            quotes_html += f"""
            <div class="evidence-quote">
                <div class="quote-text">"{quote.get('text', '')}"</div>
                <div class="quote-meta">
                    <strong>{quote.get('author', 'Anonymous')}</strong> · {quote.get('source', 'Google Reviews')} · {quote.get('date', '2024')}
                    <br>⭐ {quote.get('rating', 5)}/10
                </div>
                <a href="{review_url}" target="_blank" class="review-link">
                    🔗 Original-Review ansehen
                </a>
            </div>
            """

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{insight.get('title', '')}</div>
            <div class="insight-text">{insight.get('text', '')}</div>
            <div class="evidence-section">
                <div class="evidence-header">📝 Beweis aus {evidence.get('mentions', 0)} Gästebewertungen:</div>
                {quotes_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("<h2 style='margin-top: 3rem; margin-bottom: 1.5rem;'>✅ Handlungsempfehlungen</h2>", unsafe_allow_html=True)

    recommendations = analysis.get("recommendations", {})

    action_categories = [
        ("immediate", "🚨 Sofort umsetzen", "action-immediate"),
        ("short_term", "📅 Kurzfristig (1-3 Monate)", "action-short"),
        ("long_term", "🎯 Langfristig (6+ Monate)", "action-long")
    ]

    for key, title, css_class in action_categories:
        actions = recommendations.get(key, [])
        if actions:
            actions_html = ""
            for action in actions:
                actions_html += f"""
                <div class="action-item">
                    <div class="action-text">{action.get('action', '')}</div>
                    <div class="action-evidence">💡 {action.get('evidence', '')}</div>
                </div>
                """

            st.markdown(f"""
            <div class="action-group">
                <div class="action-group-title {css_class}">{title}</div>
                {actions_html}
            </div>
            """, unsafe_allow_html=True)

    # Reset button
    st.write("")
    if st.button("🔄 Neues Hotel analysieren"):
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()
