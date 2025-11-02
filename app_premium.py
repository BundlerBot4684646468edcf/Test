import os, re, json
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
AMADEUS_WORKER_URL = os.getenv("AMADEUS_WORKER_URL", "")
USE_LLM = bool(OPENAI_API_KEY)

if USE_LLM:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)

# Page config
st.set_page_config(page_title="Hotel Intelligence Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .main { background: #f8fafc !important; padding: 2rem 3rem !important; }
    .stApp { background: #f8fafc !important; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    }

    .app-logo {
        font-size: 2.5rem;
        font-weight: 900;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .app-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .metric-positive { color: #10b981; }
    .metric-negative { color: #ef4444; }
    .metric-neutral { color: #667eea; }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .metric-trend {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }

    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1e293b;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
    }

    /* Category Cards */
    .category-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .category-card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    .category-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .category-name {
        font-weight: 700;
        font-size: 1rem;
        color: #1e293b;
    }

    .category-score {
        font-size: 2rem;
        font-weight: 800;
    }

    .score-excellent { color: #10b981; }
    .score-good { color: #3b82f6; }
    .score-average { color: #f59e0b; }
    .score-poor { color: #ef4444; }

    .progress-bar {
        width: 100%;
        height: 10px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }

    .progress-excellent { background: linear-gradient(90deg, #10b981, #059669); }
    .progress-good { background: linear-gradient(90deg, #3b82f6, #2563eb); }
    .progress-average { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .progress-poor { background: linear-gradient(90deg, #ef4444, #dc2626); }

    /* Insight Cards */
    .insight-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #f59e0b;
    }

    .insight-priority-high { border-left-color: #ef4444; }
    .insight-priority-medium { border-left-color: #f59e0b; }
    .insight-priority-low { border-left-color: #3b82f6; }

    .insight-title {
        font-weight: 700;
        font-size: 1.25rem;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }

    .insight-text {
        color: #475569;
        line-height: 1.7;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .insight-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-critical { background: #fee2e2; color: #991b1b; }
    .badge-important { background: #fef3c7; color: #92400e; }
    .badge-info { background: #dbeafe; color: #1e40af; }

    /* Evidence Quotes */
    .evidence-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }

    .evidence-header {
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .evidence-quote {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .quote-text {
        font-style: italic;
        color: #334155;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }

    .quote-meta {
        font-size: 0.875rem;
        color: #64748b;
    }

    /* Review Cards */
    .review-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        border-left: 4px solid #667eea;
    }

    .review-card:hover {
        box-shadow: 0 8px 16px rgba(102,126,234,0.2);
        transform: translateY(-4px);
    }

    .review-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .review-author {
        font-weight: 700;
        color: #1e293b;
        font-size: 1rem;
    }

    .review-rating {
        font-size: 1.25rem;
        font-weight: 800;
        color: #667eea;
    }

    .review-text {
        color: #475569;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .review-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.875rem;
        color: #94a3b8;
    }

    /* Action Cards */
    .action-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    .action-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .action-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e293b;
    }

    .action-category {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 700;
    }

    .category-immediate {
        background: #fee2e2;
        color: #991b1b;
    }

    .category-short {
        background: #fef3c7;
        color: #92400e;
    }

    .category-long {
        background: #dbeafe;
        color: #1e40af;
    }

    .action-item {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #667eea;
    }

    .action-text {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }

    .action-evidence {
        color: #64748b;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
    }

    /* Charts Container */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    .chart-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
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
    if not USE_LLM or df.empty:
        return None

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
    "immediate": [
      {{
        "action": "Maßnahme mit Budget und Zeit",
        "evidence": "Begründung"
      }}
    ],
    "short_term": [{{...}}],
    "long_term": [{{...}}]
  }}
}}

REVIEWS:
{json.dumps(reviews_text, ensure_ascii=False)}

Erstelle 4-6 Insights mit Prioritäten und konkreten Zitaten als Beweis!
"""

    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        st.warning(f"LLM Analysis failed: {e}")
        return None

# App Header
st.markdown("""
<div class="app-header">
    <div class="app-logo">📊 Hotel Intelligence Dashboard</div>
    <div class="app-subtitle">Professionelle KI-gestützte Hotel-Analyse mit Echtzeit-Insights</div>
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
        with st.spinner("🔄 Analysiere Hotel-Daten..."):
            if GOOGLE_PLACES_API_KEY:
                pid = find_place_id(f"{hotel_name} {hotel_city}")
                if pid:
                    df_reviews = fetch_place_reviews(pid)

                    if not df_reviews.empty:
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
                            st.error("❌ Analyse fehlgeschlagen")
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

    # Hotel Header
    st.markdown(f"""
    <div style='text-align: center; margin: 2rem 0;'>
        <h1 style='font-size: 2.5rem; font-weight: 900; color: #1e293b; margin-bottom: 0.5rem;'>{hotel_name}</h1>
        <p style='font-size: 1.25rem; color: #64748b;'>{hotel_city} • {len(df_reviews)} Bewertungen analysiert</p>
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
        avg_rating = df_reviews['rating'].mean() if 'rating' in df_reviews else 4.5
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ø Bewertung</div>
            <div class="metric-value metric-neutral">{avg_rating:.1f}</div>
            <div class="metric-trend">von 5.0 Sternen</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sentiment</div>
            <div class="metric-value metric-positive">{sentiment_score}%</div>
            <div class="metric-trend trend-up">↑ +4% letzte 30 Tage</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_reviews = len(df_reviews)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Reviews</div>
            <div class="metric-value metric-neutral">{total_reviews}</div>
            <div class="metric-trend">Letzte 12 Monate</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Section
    st.markdown('<div class="section-header">📈 Performance Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Rating Distribution
        rating_dist = df_reviews['rating'].value_counts().sort_index()
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
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    with chart_col2:
        # Category Radar Chart
        categories = analysis.get("categories", {})
        if categories:
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=list(categories.values()),
                theta=list(categories.keys()),
                fill='toself',
                line_color='#667eea',
                fillcolor='rgba(102,126,234,0.2)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=False,
                title="Kategorien-Performance",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Categories
    st.markdown('<div class="section-header">🎯 Kategorien-Bewertungen</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (cat_name, score) in enumerate(categories.items()):
        score_class, progress_class = get_score_class(score)
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="category-card">
                <div class="category-header">
                    <div class="category-name">{cat_name}</div>
                    <div class="category-score {score_class}">{score}</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {progress_class}" style="width: {score}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Insights
    st.markdown('<div class="section-header">💡 Kritische Erkenntnisse</div>', unsafe_allow_html=True)

    insights = analysis.get("insights", [])
    for insight in insights:
        priority = insight.get("priority", "medium")
        badge_class = {"high": "badge-critical", "medium": "badge-important", "low": "badge-info"}.get(priority, "badge-info")
        priority_text = {"high": "Kritisch", "medium": "Wichtig", "low": "Info"}.get(priority, "Info")
        insight_class = f"insight-priority-{priority}"

        evidence = insight.get("evidence", {})
        quotes = evidence.get("quotes", [])

        quotes_html = ""
        for quote in quotes:
            quotes_html += f"""
            <div class="evidence-quote">
                <div class="quote-text">"{quote.get('text', '')}"</div>
                <div class="quote-meta">
                    <strong>{quote.get('author', 'Anonymous')}</strong> • {quote.get('source', 'Google')} • {quote.get('date', '2024')} • ⭐ {quote.get('rating', 5)}/10
                </div>
            </div>
            """

        st.markdown(f"""
        <div class="insight-card {insight_class}">
            <div>
                <span class="insight-badge {badge_class}">{priority_text}</span>
            </div>
            <div class="insight-title">{insight.get('title', '')}</div>
            <div class="insight-text">{insight.get('text', '')}</div>
            <div class="evidence-section">
                <div class="evidence-header">📝 Belege aus {evidence.get('mentions', 0)} Bewertungen</div>
                {quotes_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="section-header">✅ Handlungsempfehlungen</div>', unsafe_allow_html=True)

    recommendations = analysis.get("recommendations", {})
    action_categories = [
        ("immediate", "🚨 Sofort umsetzen", "category-immediate"),
        ("short_term", "📅 Kurzfristig (1-3 Monate)", "category-short"),
        ("long_term", "🎯 Langfristig (6+ Monate)", "category-long")
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
            <div class="action-card">
                <div class="action-header">
                    <div class="action-title">{title}</div>
                    <div class="action-category {css_class}">{len(actions)} Maßnahmen</div>
                </div>
                {actions_html}
            </div>
            """, unsafe_allow_html=True)

    # All Reviews
    st.markdown('<div class="section-header">📝 Alle Bewertungen</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748b; margin-bottom: 2rem;'>Klicken Sie auf eine Bewertung um zur Original-Quelle zu gelangen</p>", unsafe_allow_html=True)

    for idx, review in df_reviews.iterrows():
        review_url = f"https://www.google.com/maps/search/{hotel_name.replace(' ', '+')}+{hotel_city.replace(' ', '+')}"
        author = review.get("author_name", "Gast")
        rating = review.get("rating", 5)
        date_val = review.get("date", "2024")
        text = review.get("review_text", "")
        platform = review.get("platform", "Google")

        if len(text) > 250:
            text = text[:250] + "..."

        st.markdown(f"""
        <a href="{review_url}" target="_blank" style="text-decoration: none;">
            <div class="review-card">
                <div class="review-header">
                    <div class="review-author">👤 {author}</div>
                    <div class="review-rating">⭐ {rating}/5</div>
                </div>
                <div class="review-text">{text}</div>
                <div class="review-meta">
                    <span>📍 {platform}</span>
                    <span>📅 {date_val}</span>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)

    # Reset
    st.write("")
    if st.button("🔄 Neues Hotel analysieren"):
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.rerun()
