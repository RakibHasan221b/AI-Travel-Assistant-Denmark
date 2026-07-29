"""Phase 12 — Explore page: pgvector semantic search (Phase 6) over real
Copenhagen places, enriched with quality score (9), vibe cluster (7), and
AI-grounded summary + sources (8) when available."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_embed_model, query

st.set_page_config(page_title="Explore — AI Denmark Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Explore Copenhagen places")

with st.form("search_form"):
    col1, col2, col3 = st.columns([3, 1, 1])
    search_query = col1.text_input("What are you looking for?", placeholder="cozy quiet cafe good for working")
    category = col2.selectbox("Category", ["", "restaurant", "cafe", "hotel", "landmark", "bar"])
    neighborhood = col3.text_input("Neighborhood (optional)", placeholder="Vesterbro")
    submitted = st.form_submit_button("Search", type="primary")

if submitted and search_query.strip():
    model = get_embed_model()
    query_embedding = model.encode(search_query)

    sql = """
        SELECT p.place_id, p.name, p.category, p.neighborhood, p.opening_hours,
               1 - (p.embedding <=> %(qvec)s) AS similarity,
               m.predicted_value AS quality_score,
               c.label AS cluster_label
        FROM places p
        LEFT JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
        LEFT JOIN place_clusters pc ON pc.place_id = p.place_id
        LEFT JOIN clusters c ON c.cluster_id = pc.cluster_id
        WHERE p.embedding IS NOT NULL
    """
    params = {"qvec": query_embedding, "limit": 15}
    if category:
        sql += " AND p.category = %(category)s"
        params["category"] = category
    if neighborhood.strip():
        sql += " AND p.neighborhood ILIKE %(neighborhood)s"
        params["neighborhood"] = f"%{neighborhood.strip()}%"
    sql += " ORDER BY p.embedding <=> %(qvec)s LIMIT %(limit)s;"

    results = query(sql, params)

    if not results:
        st.info("No matching places found. Try a broader query or fewer filters.")
    else:
        for r in results:
            with st.container(border=True):
                header_col, score_col = st.columns([4, 1])
                header_col.markdown(
                    f"**{r['name']}** — {r['category']}, {r['neighborhood'] or 'unknown area'} "
                    f"· similarity {r['similarity']:.2f}"
                )
                if r["quality_score"] is not None:
                    score_col.metric("Quality", f"{r['quality_score']:.0f}/100")

                meta_bits = []
                if r["cluster_label"]:
                    meta_bits.append(f"Vibe: *{r['cluster_label']}*")
                if r["opening_hours"]:
                    meta_bits.append(f"Hours: {r['opening_hours']}")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))

                summary_rows = query(
                    "SELECT summary_text, sources FROM ai_summaries WHERE place_id = %s "
                    "ORDER BY generated_at DESC LIMIT 1;",
                    (r["place_id"],),
                )
                if summary_rows:
                    st.write(summary_rows[0]["summary_text"])
                    source_urls = [s.get("source_url") for s in summary_rows[0]["sources"] if s.get("source_url")]
                    if source_urls:
                        st.caption("Sources: " + ", ".join(source_urls))
elif submitted:
    st.warning("Type something to search for.")
