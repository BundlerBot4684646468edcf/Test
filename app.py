import os, re, json, math
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----------------- Setup -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
USE_LLM = bool(OPENAI_API_KEY)

if USE_LLM:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="AI Hotel Reputation — MVP", layout="wide")
st.title("🏨 AI Hotel Reputation — MVP")
st.caption("Hotel name → reviews → language AI + ML → clusters, trends & action plan (DE/IT/EN).")

# ----------------- Helpers -----------------
POS = {"great","amazing","love","excellent","freundlich","gentile","eccellente","perfetto","clean","modern","spacious","cozy","super"}
NEG = {"broken","dirty","cold","hot","slow","unfriendly","unfreundlich","rumore","loud","schmutzig","scarso","pessimo","old","outdated","queue","wait","noisy"}

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

# ----------------- LLM analysis -----------------
def llm_parse_reviews(df: pd.DataFrame)->pd.DataFrame:
    """Ask LLM to structure reviews; fallback to heuristics if unavailable/fails."""
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
                "issues": [],
                "aspects": []
            })
        return pd.DataFrame(out)

    # keep batches small to control tokens
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": str(r["date"]),
            "lang": (r.get("language") or detect_lang(r["review_text"])),
            "rating": int(pd.to_numeric(r.get("rating", 0), errors="coerce") or 0),
            "text": clean_text(r["review_text"]).replace('"', "'")
        })
    system = "You analyze multilingual hotel reviews and respond with STRICT JSON ONLY. No prose."
    user = f"""
For each item return:
- detected_language (DE|IT|EN),
- overall_sentiment (-2..2),
- aspects: list of {{name, sentiment(-2..2), quote}},
- issues: list of short phrases.
FORMAT: {{"items":[{{"date":"YYYY-MM-DD","detected_language":"DE|IT|EN","overall_sentiment":-2,"aspects":[{{"name":"...","sentiment":-2,"quote":"..."}}],"issues":["..."]}}]}}

ITEMS:
{json.dumps(rows, ensure_ascii=False)}
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
                "issues": it.get("issues", []),
                "aspects": it.get("aspects", [])
            })
        return pd.DataFrame(out)
    except Exception as e:
        st.warning(f"LLM parse failed → fallback heuristic. ({e})")
        return llm_parse_reviews(pd.DataFrame([], columns=df.columns))

def llm_action_plan(clusters_df: pd.DataFrame, df_struct: pd.DataFrame, top_k=3)->dict:
    if not USE_LLM or clusters_df.empty:
        return {"summary":"(LLM disabled)","ops_actions":[],"marketing_actions":[],"copy":{}}
    top = clusters_df.head(top_k)
    bundle = []
    for _, row in top.iterrows():
        quotes = df_struct[df_struct["cluster"]==row["cluster"]]["review_text"].head(3).tolist()
        bundle.append({
            "cluster": int(row["cluster"]),
            "impact": float(round(row["impact"],4)),
            "avg_sentiment": float(round(row["avg_sentiment_w"],3)),
            "freshness": float(round(row["freshness"],3)),
            "examples": quotes
        })
    system = "You are a hotel operations & marketing advisor. Reply STRICT JSON ONLY."
    user = f"""
Hotel type: 4* wellness (DE/IT/EN guests). Based on the clusters below, propose concrete actions and ready-to-use copy.
Return JSON:
{{
 "summary":"1-2 lines",
 "ops_actions":[{{"cluster":0,"fix":"...","owner":"Front Office|Maintenance|Kitchen","ETA":"7 days","metric":"..."}}],
 "marketing_actions":[{{"cluster":0,"audience":"Returning couples","idea":"...","channel":"Email|SMS|Social","KPI":"..."}}],
 "copy":{{
   "DE":{{"subject":["...","..."],"email_short":"...","sms":"..."}},
   "IT":{{"subject":["...","..."],"email_short":"...","sms":"..."}},
   "EN":{{"subject":["...","..."],"email_short":"...","sms":"..."}}
 }}
}}

CLUSTERS:
{json.dumps(bundle, ensure_ascii=False)}
"""
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        st.warning(f"Action-plan parse failed: {e}")
        return {"summary":"(parse error)","ops_actions":[],"marketing_actions":[]}

# ----------------- ML clustering & scoring -----------------
def build_clusters(df_struct: pd.DataFrame, n_clusters:int=12):
    if df_struct.empty:
        return df_struct.assign(cluster=-1), None, None
    corpus = []
    for _, r in df_struct.iterrows():
        issues = " ".join(r.get("issues", [])) if isinstance(r.get("issues", []), list) else ""
        corpus.append(clean_text(f"{r['review_text']} {issues}"))
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
n_clusters = st.sidebar.slider("KMeans clusters", 6, 20, 12, 1)

st.subheader("1) Input")
col1, col2 = st.columns([2,1])
with col1:
    hotel_query = st.text_input("Hotel name (Google Places)", placeholder="e.g., ADLER Spa Resort Dolomiti, Ortisei")
with col2:
    st.write(" ")
    fetch_btn = st.button("🔎 Fetch Reviews")

uploaded = st.file_uploader("…or upload reviews CSV (date,platform,language,rating,review_text)", type=["csv"])

frames = []
if fetch_btn and hotel_query.strip():
    if not GOOGLE_PLACES_API_KEY:
        st.error("Missing GOOGLE_PLACES_API_KEY in .env")
    else:
        with st.spinner("Finding place…"):
            pid = find_place_id(hotel_query.strip())
        if not pid:
            st.error("Place not found. Try a more specific hotel string.")
        else:
            with st.spinner("Fetching reviews…"):
                df_g = fetch_place_reviews(pid)
            if df_g.empty:
                st.warning("No reviews returned (Places often exposes ~5 latest). Consider uploading a CSV too.")
            else:
                frames.append(df_g)

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

    st.subheader("2) Analyze")
    if st.button("🚀 Run Analysis"):
        with st.spinner("Language AI: understanding reviews…"):
            df_struct = llm_parse_reviews(df_raw)
        df_struct["language"] = df_struct["language"].fillna("").apply(lambda x: x if x else "EN")
        df_struct["overall_sentiment"] = df_struct["overall_sentiment"].fillna(0)

        with st.spinner("ML: clustering topics…"):
            df_struct, tfidf, kmeans = build_clusters(df_struct, n_clusters=n_clusters)

        with st.spinner("Scoring: impact & freshness…"):
            clusters = aggregate_clusters(df_struct, half_life_days=half_life_days)

        # KPIs
        st.subheader("3) Results")
        c1,c2,c3,c4 = st.columns(4)
        avg_rating = pd.to_numeric(df_struct["rating"], errors="coerce").dropna().mean()
        neg_share = (df_struct["overall_sentiment"]<0).mean() if len(df_struct) else 0
        top_issue = clusters.iloc[0]["cluster"] if not clusters.empty else "—"
        top_sent = clusters.iloc[0]["avg_sentiment_w"] if not clusters.empty else 0
        with c1: st.metric("Avg rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "—")
        with c2: st.metric("% negative", f"{neg_share*100:.1f}%")
        with c3: st.metric("Top topic (cluster id)", str(top_issue))
        with c4: st.metric("Top topic mood", f"{top_sent:.2f}")

        # Charts
        df_struct["date_dt"] = pd.to_datetime(df_struct["date"], errors="coerce")
        daily = df_struct.dropna(subset=["date_dt"]).groupby(pd.Grouper(key="date_dt", freq="W")).agg(
            pos=("overall_sentiment", lambda s: (s>0).sum()),
            neg=("overall_sentiment", lambda s: (s<0).sum())
        ).reset_index()
        if not daily.empty:
            st.plotly_chart(px.line(daily, x="date_dt", y=["pos","neg"], title="Weekly mood (pos vs neg)"), use_container_width=True)

        if not clusters.empty:
            st.plotly_chart(px.bar(clusters.head(12), x="cluster", y="impact",
                                   hover_data=["avg_sentiment_w","freshness","frequency_norm"],
                                   title="Top topics by impact"),
                            use_container_width=True)
            st.write("**Topic table**")
            st.dataframe(clusters, use_container_width=True)

        # Action plan
        st.subheader("4) Action Plan (DE/IT/EN)")
        if USE_LLM:
            if st.button("📝 Generate Actions"):
                with st.spinner("Creating ops & marketing plan…"):
                    plan = llm_action_plan(clusters, df_struct, top_k=3)
                st.json(plan)
        else:
            st.info("Set OPENAI_API_KEY in .env to enable multilingual action plans.")

        # Export
        st.subheader("5) Export")
        enriched = df_struct.copy()
        enriched["weight"] = enriched["date"].apply(lambda d: half_life_weight(d, half_life_days))
        st.download_button("⬇️ Reviews (analyzed).csv", enriched.to_csv(index=False).encode("utf-8"),
                           file_name=f"reviews_analyzed_{date.today().isoformat()}.csv", mime="text/csv")
        if not clusters.empty:
            st.download_button("⬇️ Topics (clusters).csv", clusters.to_csv(index=False).encode("utf-8"),
                               file_name=f"clusters_{date.today().isoformat()}.csv", mime="text/csv")
else:
    st.info("Enter a hotel name and/or upload a CSV to begin.")

st.markdown("---")
st.caption("MVP • Hybrid: Language AI (LLM) + ML • Freshness-weighted • Actionable outputs")
