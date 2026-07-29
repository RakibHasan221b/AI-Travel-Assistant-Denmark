"""Phase 12 — Stats Dashboard: real aggregate queries over Phases 7/9/10's
stored output. Every number here is computed from Postgres, not invented."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import query

st.set_page_config(page_title="Stats — AI Denmark Explorer", page_icon="📊", layout="wide")
st.title("📊 Stats Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Avg quality score by neighborhood")
    rows = query(
        """
        SELECT COALESCE(p.neighborhood, 'unassigned') AS neighborhood,
               AVG(m.predicted_value) AS avg_quality_score, COUNT(*) AS n_places
        FROM places p
        JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
        GROUP BY p.neighborhood
        HAVING COUNT(*) >= 10
        ORDER BY avg_quality_score DESC;
        """
    )
    df = pd.DataFrame(rows).set_index("neighborhood")
    st.bar_chart(df["avg_quality_score"])
    st.caption("Neighborhoods with fewer than 10 places omitted to avoid noisy averages.")

with col2:
    st.subheader("Avg quality score by category")
    rows = query(
        """
        SELECT p.category, AVG(m.predicted_value) AS avg_quality_score, COUNT(*) AS n_places
        FROM places p
        JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
        GROUP BY p.category
        ORDER BY avg_quality_score DESC;
        """
    )
    df = pd.DataFrame(rows).set_index("category")
    st.bar_chart(df["avg_quality_score"])

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Vibe clusters (Phase 7)")
    rows = query(
        """
        SELECT c.label, COUNT(*) AS n_places
        FROM clusters c
        JOIN place_clusters pc ON pc.cluster_id = c.cluster_id
        GROUP BY c.label
        ORDER BY n_places DESC;
        """
    )
    df = pd.DataFrame(rows).set_index("label")
    st.bar_chart(df["n_places"])

with col4:
    st.subheader("Rated aspects (avg score, 1-5)")
    rows = query(
        """
        SELECT aspect_category, AVG(avg_score) AS mean_score, SUM(num_mentions) AS total_mentions
        FROM aggregated_sentiment
        GROUP BY aspect_category
        ORDER BY mean_score DESC;
        """
    )
    df = pd.DataFrame(rows).set_index("aspect_category")
    st.bar_chart(df["mean_score"])
    st.caption(f"Based on {int(df['total_mentions'].sum()) if not df.empty else 0} total rated mentions.")

st.divider()

st.subheader("Weather-driven outdoor interest forecast (Phase 10)")
rows = query(
    """
    SELECT vtf.forecast_date, p.category, AVG(vtf.predicted_interest_score) AS avg_interest
    FROM visit_time_forecast vtf
    JOIN places p ON p.place_id = vtf.place_id
    GROUP BY vtf.forecast_date, p.category
    ORDER BY vtf.forecast_date;
    """
)
if rows:
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="forecast_date", columns="category", values="avg_interest")
    st.line_chart(pivot)
    st.caption("Outdoor Interest Index (0-100) per category, current ~7-day forecast window.")
else:
    st.info("No forecast data in range — the forecast window may need refreshing (re-run pipeline/timeseries/forecast_interest.py).")
