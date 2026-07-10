---
tags: [product]
status: MVP
---

# Product Overview — AI Hotel Reputation

> Hotel name → reviews → language AI + ML → clusters, trends & action plan (DE/IT/EN).

## What it does (current MVP)
1. **Find hotel** via Google Places (`GOOGLE_PLACES_API_KEY`)
2. **Fetch reviews** and clean the text
3. **Detect language** (DE / IT / EN) and score sentiment (heuristic, LLM-assisted when `OPENAI_API_KEY` is set)
4. **Cluster topics** with TF-IDF + KMeans
5. **Weight by recency** (half-life 180 days) and show **trends** (Plotly)
6. Output a prioritized **action plan** for the hotel

## Stack
- Streamlit UI (`app.py` in repo root)
- pandas / numpy / scikit-learn / plotly
- Google Places API, optional OpenAI

## Roadmap
| Horizon | Item | Notes |
|---|---|---|
| Now | Validate clusters with pilot hotels | see [[05-Projects/Pilot Program]] |
| Next | More review sources (Booking, TripAdvisor) | needs sourcing decision → [[08-Decisions/Decisions Index]] |
| Later | Auto-generated weekly action-plan email | |

## Related
- [[07-Knowledge/Dev Setup]]
- [[06-Customers/Customers Index]]
