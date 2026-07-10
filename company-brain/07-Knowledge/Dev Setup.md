---
tags: [knowledge, engineering]
---

# Dev Setup

How to run the AI Hotel Reputation MVP locally.

## Prerequisites
- Python 3.11+
- API keys in a `.env` file at the repo root:
  ```
  GOOGLE_PLACES_API_KEY=...   # required — hotel lookup & reviews
  OPENAI_API_KEY=...          # optional — enables LLM features
  ```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints, enter a hotel name, and run the analysis.

## Notes
- Without `OPENAI_API_KEY` the app falls back to heuristic sentiment (keyword lists in `app.py`).
- Review recency weighting uses a 180-day half-life.

## Related
- [[03-Product/Product Overview]]
