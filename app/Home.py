"""Phase 12 — Streamlit entry point. See docs/technique_map.md for the full
technique-to-phase mapping this app sits on top of."""

import streamlit as st
from common import query

st.set_page_config(page_title="AI Denmark Explorer", page_icon="🧭", layout="wide")

st.title("🧭 AI Denmark Explorer")
st.caption("A real Copenhagen place-discovery pilot — every number on this site is computed from real data, not invented.")

st.markdown(
    """
Use the sidebar to navigate:

- **Explore** — semantic search over 1,468 real Copenhagen places (pgvector), with quality scores, vibe clusters, and AI-grounded summaries where available.
- **Trip Planner** — a CrewAI multi-agent crew (Place Scout, Concierge) plans a real recommendation from your request.
- **Stats Dashboard** — aggregate numbers computed straight from Postgres: quality scores by neighborhood, cluster sizes, sentiment distribution, weather-driven forecasts.
"""
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
counts = query(
    """
    SELECT
        (SELECT COUNT(*) FROM places) AS places,
        (SELECT COUNT(*) FROM clusters) AS clusters,
        (SELECT COUNT(*) FROM ai_summaries) AS summaries,
        (SELECT COUNT(*) FROM ml_predictions WHERE target = 'quality_score') AS scored
    """
)[0]
col1.metric("Places", f"{counts['places']:,}")
col2.metric("Vibe clusters", counts["clusters"])
col3.metric("AI summaries", counts["summaries"])
col4.metric("Quality-scored places", f"{counts['scored']:,}")
