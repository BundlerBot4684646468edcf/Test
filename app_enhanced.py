import os, re, json, math
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="🏨 Hotel Intelligence Hub — AI Sentiment Analysis", layout="wide")
st.title("🏨 Hotel Intelligence Hub")
st.caption("Professional AI sentiment analysis for hospitality • Multi-source • Multilingual (DE/IT/EN) • Actionable insights")

# ----------------- Named Categories -----------------
CATEGORY_KEYWORDS = {
    "Rooms & Cleanliness": ["room", "zimmer", "camera", "clean", "sauber", "pulito", "bed", "bett", "letto", "bathroom", "badezimmer", "bagno", "shower", "dusche", "doccia", "spacious", "geräumig", "spazioso", "comfortable", "komfortabel", "comodo"],
    "Service & Staff": ["staff", "personal", "personale", "service", "reception", "rezeption", "ricevimento", "friendly", "freundlich", "gentile", "helpful", "hilfsbereit", "cortese", "professional", "professionell", "professionale", "attention", "aufmerksamkeit", "attenzione"],
    "Food & Breakfast": ["food", "essen", "cibo", "breakfast", "frühstück", "colazione", "dinner", "abendessen", "cena", "restaurant", "coffee", "kaffee", "caffè", "buffet", "quality", "qualität", "qualità", "variety", "vielfalt", "varietà", "delicious", "lecker", "delizioso"],
    "Location & View": ["location", "lage", "posizione", "view", "aussicht", "vista", "central", "zentral", "centrale", "proximity", "nähe", "vicinanza", "transport", "verkehr", "trasporto", "parking", "parkplatz", "parcheggio", "quiet", "ruhig", "tranquillo"],
    "Spa & Wellness": ["spa", "wellness", "pool", "schwimmbad", "piscina", "sauna", "massage", "massaggio", "relax", "entspannung", "rilassamento", "treatment", "behandlung", "trattamento", "gym", "fitness", "sport"],
    "Price & Value": ["price", "preis", "prezzo", "value", "wert", "valore", "expensive", "teuer", "costoso", "cheap", "günstig", "economico", "worth", "lohnt", "vale", "money", "geld", "soldi", "quality", "qualität", "qualità"]
}

# ----------------- Helpers -----------------
POS = {"great","amazing","love","excellent","freundlich","gentile","eccellente","perfetto","clean","modern","spacious","cozy","super","wonderful","fantastic","beautiful"}
NEG = {"broken","dirty","cold","hot","slow","unfriendly","unfreundlich","rumore","loud","schmutzig","scarso","pessimo","old","outdated","queue","wait","noisy","terrible","awful","disappointing"}

def clean_text(s:str)->str:
    if not isinstance(s, str): return ""
    s = re.sub(r"\s+", " ", s.strip())
    return s

def detect_lang(text:str)->str:
    try:
        l = detect(text or "")
    except Exception:
        l = "en"
    l = l.upper()[:2]
    return l if l in {"DE","IT","EN"} else "EN"

def heuristic_sentiment(text:str)->int:
    t = (text or "").lower()
    pos = any(w in t for w in POS)
    neg = any(w in t for w in NEG)
    return 1 if (pos and not neg) else (-1 if (neg and not pos) else 0)

def half_life_weight(d: str, half_life_days=180):
    try:
        dt = pd.to_datetime(d).date()
    except Exception:
        return 1.0
    age = (date.today() - dt).days
    age = max(age, 0)
    return 0.5 ** (age / half_life_days)

# ----------------- Amadeus Integration -----------------
def fetch_amadeus_token()->str:
    """Fetch Amadeus access token from Cloudflare Worker"""
    if not AMADEUS_WORKER_URL:
        return ""
    try:
        r = requests.get(AMADEUS_WORKER_URL, timeout=10)
        r.raise_for_status()
        return r.json().get("access_token", "")
    except Exception as e:
        st.warning(f"Amadeus token fetch failed: {e}")
        return ""

def fetch_amadeus_hotel_data(hotel_name:str, city:str=""):
    """Fetch hotel data from Amadeus API"""
    token = fetch_amadeus_token()
    if not token:
        return None

    try:
        # Search for hotel by name
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
        params = {"cityCode": city} if city else {}

        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()

        # For now, return basic structure
        # In production, you'd search and match the hotel_name
        return {
            "hotel_name": hotel_name,
            "amadeus_data": "available",
            "source": "Amadeus Test API"
        }
    except Exception as e:
        st.warning(f"Amadeus hotel data fetch failed: {e}")
        return None

# ----------------- Google Places -----------------
def find_place_id(hotel_query:str)->str|None:
    if not GOOGLE_PLACES_API_KEY: return None
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {"input": hotel_query, "inputtype":"textquery", "fields":"place_id,name,formatted_address", "key": GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    c = r.json().get("candidates", [])
    return c[0]["place_id"] if c else None

def fetch_place_reviews(place_id:str, max_reviews:int=30)->pd.DataFrame:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id":place_id, "fields":"reviews,name,formatted_address,url,user_ratings_total,rating", "key":GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    result = r.json().get("result", {})
    reviews = result.get("reviews", []) or []
    rows = []
    for rv in reviews[:max_reviews]:
        ts = rv.get("time")
        d = datetime.utcfromtimestamp(ts).date().isoformat() if isinstance(ts,(int,float)) else None
        rows.append({
            "date": d,
            "platform": "Google",
            "language": (rv.get("language") or "").upper()[:2] if rv.get("language") else "",
            "rating": rv.get("rating"),
            "review_text": rv.get("text"),
            "source_url": result.get("url")
        })
    return pd.DataFrame(rows)

# ----------------- Enhanced LLM analysis with Emotions & Categories -----------------
def llm_parse_reviews_enhanced(df: pd.DataFrame)->pd.DataFrame:
    """Enhanced LLM analysis with emotions and named categories"""
    if not USE_LLM or df.empty:
        out = []
        for _, r in df.iterrows():
            lang = r.get("language") or detect_lang(r["review_text"])
            out.append({
                "date": r["date"],
                "platform": r["platform"],
                "language": lang,
                "rating": r.get("rating", None),
                "review_text": r["review_text"],
                "overall_sentiment": heuristic_sentiment(r["review_text"]),
                "emotion": "Neutral",
                "category": "General",
                "issues": [],
                "aspects": []
            })
        return pd.DataFrame(out)

    # Process reviews with enhanced prompt
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": str(r["date"]),
            "lang": (r.get("language") or detect_lang(r["review_text"])),
            "rating": int(pd.to_numeric(r.get("rating", 0), errors="coerce") or 0),
            "text": clean_text(r["review_text"]).replace('"', "'")
        })

    system = """You are Hotel Intelligence AI, a professional sentiment-analysis system for the hospitality industry.
You analyze multilingual hotel reviews and respond with STRICT JSON ONLY. No prose."""

    user = f"""
Analyze each review and return:
- detected_language (DE|IT|EN)
- overall_sentiment (-2 to 2, where -2=very negative, 0=neutral, 2=very positive)
- emotion (Joy|Anger|Sadness|Surprise|Neutral)
- category (Rooms & Cleanliness|Service & Staff|Food & Breakfast|Location & View|Spa & Wellness|Price & Value|General)
- aspects: list of {{name, sentiment(-2..2), quote}}
- issues: list of short problem phrases

FORMAT: {{"items":[{{"date":"YYYY-MM-DD","detected_language":"DE|IT|EN","overall_sentiment":1,"emotion":"Joy","category":"Rooms & Cleanliness","aspects":[{{"name":"room quality","sentiment":2,"quote":"amazing room"}}],"issues":[]}}]}}

ITEMS:
{json.dumps(rows[:10], ensure_ascii=False)}
"""

    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        payload = resp.choices[0].message.content.strip()
        js = json.loads(payload)
        items = js.get("items", [])
        out = []
        for i, r in enumerate(df.itertuples()):
            it = items[i] if i < len(items) else {}
            out.append({
                "date": r.date,
                "platform": r.platform,
                "language": it.get("detected_language") or (r.language or detect_lang(r.review_text)),
                "rating": r.rating,
                "review_text": r.review_text,
                "overall_sentiment": it.get("overall_sentiment", heuristic_sentiment(r.review_text)),
                "emotion": it.get("emotion", "Neutral"),
                "category": it.get("category", "General"),
                "issues": it.get("issues", []),
                "aspects": it.get("aspects", [])
            })
        return pd.DataFrame(out)
    except Exception as e:
        st.warning(f"LLM parse failed → fallback heuristic. ({e})")
        return llm_parse_reviews_enhanced(pd.DataFrame([], columns=df.columns))

def generate_structured_output(hotel_name:str, city:str, df_struct:pd.DataFrame, clusters_df:pd.DataFrame)->dict:
    """Generate structured JSON output matching the system prompt format"""
    if df_struct.empty:
        return {}

    # Calculate overall sentiment score (0-1 scale)
    avg_sentiment_raw = df_struct["overall_sentiment"].mean()
    overall_sentiment_score = (avg_sentiment_raw + 2) / 4  # Convert -2..2 to 0..1

    # Build review clusters by category
    review_clusters = {}
    for category in CATEGORY_KEYWORDS.keys():
        cat_reviews = df_struct[df_struct["category"] == category]
        if not cat_reviews.empty:
            cat_score = (cat_reviews["overall_sentiment"].mean() + 2) / 4  # 0-1 scale
            # Get most common issues
            all_issues = []
            for issues_list in cat_reviews["issues"]:
                if isinstance(issues_list, list):
                    all_issues.extend(issues_list)
            top_issues = pd.Series(all_issues).value_counts().head(3).index.tolist() if all_issues else []

            review_clusters[category] = {
                "score": round(cat_score, 2),
                "summary": f"{len(cat_reviews)} reviews analyzed. " + (f"Common issues: {', '.join(top_issues[:2])}" if top_issues else "Generally positive feedback.")
            }

    # Key issues across all categories
    all_issues = []
    for issues_list in df_struct["issues"]:
        if isinstance(issues_list, list):
            all_issues.extend(issues_list)
    key_issues = pd.Series(all_issues).value_counts().head(5).index.tolist() if all_issues else []

    # Recommended actions (simplified)
    recommended_actions = []
    if not clusters_df.empty:
        for _, cluster in clusters_df.head(3).iterrows():
            if cluster.get("avg_sentiment_w", 0) < -0.5:
                recommended_actions.append(f"Address concerns in cluster {cluster.get('cluster', 0)}: prioritize improvements")

    return {
        "hotel_name": hotel_name,
        "city": city,
        "language": df_struct["language"].mode()[0] if not df_struct.empty else "EN",
        "overall_sentiment_score": round(overall_sentiment_score, 2),
        "review_clusters": review_clusters,
        "key_issues": key_issues,
        "recommended_actions": recommended_actions if recommended_actions else ["Continue monitoring guest feedback", "Maintain current service standards"]
    }

def llm_action_plan_enhanced(clusters_df: pd.DataFrame, df_struct: pd.DataFrame, hotel_name:str, top_k=3)->dict:
    """Enhanced action plan with category awareness"""
    if not USE_LLM or clusters_df.empty:
        return {"summary":"(LLM disabled)","ops_actions":[],"marketing_actions":[],"copy":{}}

    top = clusters_df.head(top_k)
    bundle = []
    for _, row in top.iterrows():
        cluster_id = row.get("cluster", 0)
        quotes = df_struct[df_struct["cluster"]==cluster_id]["review_text"].head(3).tolist()
        categories = df_struct[df_struct["cluster"]==cluster_id]["category"].value_counts().head(1)
        main_category = categories.index[0] if not categories.empty else "General"

        bundle.append({
            "cluster": int(cluster_id),
            "category": main_category,
            "impact": float(round(row["impact"],4)),
            "avg_sentiment": float(round(row["avg_sentiment_w"],3)),
            "freshness": float(round(row["freshness"],3)),
            "examples": quotes
        })

    system = "You are a hotel operations & marketing advisor. Reply STRICT JSON ONLY."
    user = f"""
Hotel: {hotel_name}. Based on the clusters below, propose concrete actions and ready-to-use copy.
Return JSON:
{{
 "summary":"1-2 lines executive summary",
 "ops_actions":[{{"cluster":0,"category":"Rooms & Cleanliness","fix":"specific action","owner":"Front Office|Maintenance|Kitchen|Housekeeping","ETA":"7 days","metric":"target KPI"}}],
 "marketing_actions":[{{"cluster":0,"category":"Service & Staff","audience":"target segment","idea":"marketing idea","channel":"Email|SMS|Social|Website","KPI":"success metric"}}],
 "copy":{{
   "DE":{{"subject":["email subject 1","email subject 2"],"email_short":"brief email body","sms":"SMS text under 160 chars"}},
   "IT":{{"subject":["subject 1","subject 2"],"email_short":"email body","sms":"SMS text"}},
   "EN":{{"subject":["subject 1","subject 2"],"email_short":"email body","sms":"SMS text"}}
 }}
}}

CLUSTERS:
{json.dumps(bundle, ensure_ascii=False)}
"""

    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        st.warning(f"Action-plan parse failed: {e}")
        return {"summary":"(parse error)","ops_actions":[],"marketing_actions":[],"copy":{}}

# ----------------- ML clustering & scoring -----------------
def build_clusters(df_struct: pd.DataFrame, n_clusters:int=6):
    """Build clusters with category awareness"""
    if df_struct.empty:
        return df_struct.assign(cluster=-1), None, None
    corpus = []
    for _, r in df_struct.iterrows():
        issues = " ".join(r.get("issues", [])) if isinstance(r.get("issues", []), list) else ""
        category = r.get("category", "")
        corpus.append(clean_text(f"{category} {r['review_text']} {issues}"))

    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_df=0.9, min_df=2, max_features=50000)
    X = tfidf.fit_transform(corpus)
    k = min(n_clusters, max(2, X.shape[0]//2))
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)
    return df_struct.assign(cluster=labels), tfidf, kmeans

def aggregate_clusters(df_struct: pd.DataFrame, half_life_days=180):
    if df_struct.empty:
        return pd.DataFrame()
    df = df_struct.copy()
    df["weight"] = df["date"].apply(lambda d: half_life_weight(d, half_life_days))
    total_w = max(df["weight"].sum(), 1e-9)

    grp = df.groupby("cluster").apply(lambda g: pd.Series({
        "freq_weight": g["weight"].sum(),
        "avg_sentiment_w": (g["overall_sentiment"] * g["weight"]).sum() / max(g["weight"].sum(), 1e-9),
        "main_category": g["category"].mode()[0] if not g["category"].empty else "General",
        "examples": g["review_text"].head(2).tolist()
    })).reset_index()

    now_90 = pd.Timestamp.today().normalize() - pd.Timedelta(days=90)
    df["recent"] = pd.to_datetime(df["date"], errors="coerce") >= now_90
    recent_w = df[df["recent"]].groupby("cluster")["weight"].sum().rename("recent_w")
    grp = grp.merge(recent_w, on="cluster", how="left").fillna({"recent_w":0.0})

    grp["frequency_norm"] = grp["freq_weight"] / total_w
    grp["neg_weight"] = 1 + np.maximum(0, -grp["avg_sentiment_w"]) * 0.5
    grp["recency_boost"] = np.where(grp["recent_w"]>0, 1.2, 1.0)
    grp["impact"] = grp["frequency_norm"] * grp["neg_weight"] * grp["recency_boost"]
    grp["freshness"] = grp["recent_w"] / np.maximum(grp["freq_weight"], 1e-9)
    return grp.sort_values("impact", ascending=False)

# ----------------- UI -----------------
st.sidebar.header("⚙️ Settings")
half_life_days = st.sidebar.slider("Freshness half-life (days)", 90, 360, 180, 10)
n_clusters = st.sidebar.slider("Topic clusters", 4, 8, 6, 1)

if AMADEUS_WORKER_URL:
    st.sidebar.success("✅ Amadeus API Connected")
else:
    st.sidebar.info("ℹ️ Add AMADEUS_WORKER_URL to .env for hotel data")

st.subheader("1️⃣ Input")
col1, col2 = st.columns([2,1])
with col1:
    hotel_query = st.text_input("Hotel name (Google Places)", placeholder="e.g., Hotel Rosengarten Bolzano")
with col2:
    st.write(" ")
    fetch_btn = st.button("🔎 Fetch Reviews")

uploaded = st.file_uploader("…or upload reviews CSV (date,platform,language,rating,review_text)", type=["csv"])

frames = []
hotel_name = ""
city = ""

if fetch_btn and hotel_query.strip():
    if not GOOGLE_PLACES_API_KEY:
        st.error("Missing GOOGLE_PLACES_API_KEY in .env")
    else:
        with st.spinner("Finding place…"):
            pid = find_place_id(hotel_query.strip())
        if not pid:
            st.error("Place not found. Try a more specific hotel string.")
        else:
            hotel_name = hotel_query.strip()
            with st.spinner("Fetching reviews…"):
                df_g = fetch_place_reviews(pid)
            if df_g.empty:
                st.warning("No reviews returned (Places often exposes ~5 latest). Consider uploading a CSV too.")
            else:
                frames.append(df_g)

            # Fetch Amadeus data
            if AMADEUS_WORKER_URL:
                with st.spinner("Fetching hotel data from Amadeus…"):
                    amadeus_data = fetch_amadeus_hotel_data(hotel_name)
                    if amadeus_data:
                        st.success(f"✅ Amadeus data retrieved for {hotel_name}")

if uploaded:
    try:
        df_u = pd.read_csv(uploaded)
        frames.append(df_u)
    except Exception as e:
        st.error(f"CSV parse error: {e}")

if frames:
    df_raw = pd.concat(frames, ignore_index=True)
    # validate columns
    need = {"date","platform","language","rating","review_text"}
    if not need.issubset(set(df_raw.columns)):
        st.error(f"CSV must include columns: {', '.join(sorted(need))}")
        st.stop()

    df_raw["review_text"] = df_raw["review_text"].astype(str)
    st.success(f"Loaded {len(df_raw)} reviews.")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("2️⃣ Analyze")
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("🧠 AI: Understanding emotions & topics…"):
            df_struct = llm_parse_reviews_enhanced(df_raw)
        df_struct["language"] = df_struct["language"].fillna("").apply(lambda x: x if x else "EN")
        df_struct["overall_sentiment"] = df_struct["overall_sentiment"].fillna(0)
        df_struct["emotion"] = df_struct["emotion"].fillna("Neutral")
        df_struct["category"] = df_struct["category"].fillna("General")

        with st.spinner("🤖 ML: Clustering similar reviews…"):
            df_struct, tfidf, kmeans = build_clusters(df_struct, n_clusters=n_clusters)

        with st.spinner("📊 Scoring: Impact & freshness…"):
            clusters = aggregate_clusters(df_struct, half_life_days=half_life_days)

        # Generate structured output
        structured_output = generate_structured_output(hotel_name or "Hotel", city, df_struct, clusters)

        # KPIs
        st.subheader("3️⃣ Results Dashboard")
        c1,c2,c3,c4 = st.columns(4)
        avg_rating = pd.to_numeric(df_struct["rating"], errors="coerce").dropna().mean()
        neg_share = (df_struct["overall_sentiment"]<0).mean() if len(df_struct) else 0
        top_category = df_struct["category"].mode()[0] if not df_struct.empty else "—"
        overall_score = structured_output.get("overall_sentiment_score", 0)

        with c1: st.metric("Overall Sentiment", f"{overall_score:.2f}", help="0=Negative, 1=Positive")
        with c2: st.metric("Avg Rating", f"{avg_rating:.2f}/10" if pd.notna(avg_rating) else "—")
        with c3: st.metric("% Negative", f"{neg_share*100:.1f}%")
        with c4: st.metric("Top Category", top_category)

        # Emotion Distribution
        st.subheader("😊 Emotion Distribution")
        emotion_counts = df_struct["emotion"].value_counts()
        fig_emotions = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Guest Emotions",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_emotions, use_container_width=True)

        # Category Breakdown
        st.subheader("📂 Review Categories")
        col_cat1, col_cat2 = st.columns(2)

        with col_cat1:
            category_counts = df_struct["category"].value_counts()
            fig_cat = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Reviews by Category",
                labels={"x": "Category", "y": "Count"}
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_cat2:
            # Sentiment by category
            cat_sentiment = df_struct.groupby("category")["overall_sentiment"].mean().sort_values()
            fig_cat_sent = px.bar(
                x=cat_sentiment.values,
                y=cat_sentiment.index,
                orientation='h',
                title="Avg Sentiment by Category",
                labels={"x": "Sentiment", "y": "Category"},
                color=cat_sentiment.values,
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_cat_sent, use_container_width=True)

        # Time trends
        st.subheader("📈 Sentiment Trends")
        df_struct["date_dt"] = pd.to_datetime(df_struct["date"], errors="coerce")
        daily = df_struct.dropna(subset=["date_dt"]).groupby(pd.Grouper(key="date_dt", freq="W")).agg(
            pos=("overall_sentiment", lambda s: (s>0).sum()),
            neg=("overall_sentiment", lambda s: (s<0).sum()),
            neutral=("overall_sentiment", lambda s: (s==0).sum())
        ).reset_index()

        if not daily.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=daily["date_dt"], y=daily["pos"], name="Positive", line=dict(color="green")))
            fig_trend.add_trace(go.Scatter(x=daily["date_dt"], y=daily["neg"], name="Negative", line=dict(color="red")))
            fig_trend.add_trace(go.Scatter(x=daily["date_dt"], y=daily["neutral"], name="Neutral", line=dict(color="gray")))
            fig_trend.update_layout(title="Weekly Sentiment Trend", xaxis_title="Date", yaxis_title="Review Count")
            st.plotly_chart(fig_trend, use_container_width=True)

        # Impact Clusters
        if not clusters.empty:
            st.subheader("🎯 High-Impact Topics")
            clusters_display = clusters.copy()
            clusters_display["cluster_name"] = clusters_display.apply(
                lambda r: f"{r['main_category']} (#{r['cluster']})", axis=1
            )
            fig_impact = px.bar(
                clusters_display.head(8),
                x="cluster_name",
                y="impact",
                color="avg_sentiment_w",
                hover_data=["freshness","frequency_norm"],
                title="Topics by Impact Score",
                labels={"cluster_name": "Topic", "impact": "Impact Score"},
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_impact, use_container_width=True)

            st.write("**Detailed Topic Analysis**")
            st.dataframe(clusters, use_container_width=True)

        # Structured JSON Output
        st.subheader("📋 Structured Intelligence Report")
        st.json(structured_output)

        # Action Plan
        st.subheader("4️⃣ Action Plan (Multilingual)")
        if USE_LLM:
            if st.button("📝 Generate Action Plan"):
                with st.spinner("Creating operational & marketing recommendations…"):
                    plan = llm_action_plan_enhanced(clusters, df_struct, hotel_name or "Hotel", top_k=3)

                st.write("**Executive Summary**")
                st.info(plan.get("summary", "No summary available"))

                col_ops, col_mkt = st.columns(2)
                with col_ops:
                    st.write("**🔧 Operations Actions**")
                    for action in plan.get("ops_actions", []):
                        with st.expander(f"{action.get('category', 'General')} - {action.get('owner', 'Team')}"):
                            st.write(f"**Fix:** {action.get('fix', 'N/A')}")
                            st.write(f"**ETA:** {action.get('ETA', 'TBD')}")
                            st.write(f"**Metric:** {action.get('metric', 'TBD')}")

                with col_mkt:
                    st.write("**📢 Marketing Actions**")
                    for action in plan.get("marketing_actions", []):
                        with st.expander(f"{action.get('category', 'General')} - {action.get('channel', 'Channel')}"):
                            st.write(f"**Audience:** {action.get('audience', 'N/A')}")
                            st.write(f"**Idea:** {action.get('idea', 'N/A')}")
                            st.write(f"**KPI:** {action.get('KPI', 'TBD')}")

                st.write("**📧 Ready-to-Use Marketing Copy**")
                copy_data = plan.get("copy", {})
                for lang in ["DE", "IT", "EN"]:
                    if lang in copy_data:
                        with st.expander(f"🌐 {lang} Copy"):
                            st.write(f"**Subject Lines:** {', '.join(copy_data[lang].get('subject', []))}")
                            st.write(f"**Email:** {copy_data[lang].get('email_short', 'N/A')}")
                            st.write(f"**SMS:** {copy_data[lang].get('sms', 'N/A')}")
        else:
            st.info("Set OPENAI_API_KEY in .env to enable AI-powered action plans.")

        # Export
        st.subheader("5️⃣ Export Data")
        col_exp1, col_exp2, col_exp3 = st.columns(3)

        enriched = df_struct.copy()
        enriched["weight"] = enriched["date"].apply(lambda d: half_life_weight(d, half_life_days))

        with col_exp1:
            st.download_button(
                "⬇️ Reviews (analyzed).csv",
                enriched.to_csv(index=False).encode("utf-8"),
                file_name=f"reviews_analyzed_{date.today().isoformat()}.csv",
                mime="text/csv"
            )

        with col_exp2:
            if not clusters.empty:
                st.download_button(
                    "⬇️ Topics (clusters).csv",
                    clusters.to_csv(index=False).encode("utf-8"),
                    file_name=f"clusters_{date.today().isoformat()}.csv",
                    mime="text/csv"
                )

        with col_exp3:
            st.download_button(
                "⬇️ Intelligence Report (JSON)",
                json.dumps(structured_output, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"intelligence_report_{date.today().isoformat()}.json",
                mime="application/json"
            )

else:
    st.info("👆 Enter a hotel name and/or upload a CSV to begin analysis.")

st.markdown("---")
st.caption("🏨 Hotel Intelligence Hub • Powered by AI (OpenAI GPT-4) + ML (scikit-learn) • Multilingual • Data-driven insights")
