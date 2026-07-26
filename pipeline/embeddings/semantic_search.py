"""Phase 6 — semantic place search: one SQL query combining vector similarity
with structured filters (category, neighborhood). This is the concrete reason
pgvector was chosen over FAISS — FAISS can't do the filter half of this query
without a separate sync-prone system.
"""

import logging
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("semantic_search")

MODEL_NAME = "all-MiniLM-L6-v2"


def search(
    query: str,
    conn,
    model: SentenceTransformer,
    category: str | None = None,
    neighborhood: str | None = None,
    limit: int = 5,
) -> list[dict]:
    query_embedding = model.encode(query)

    sql = """
        SELECT name, category, neighborhood, opening_hours,
               1 - (embedding <=> %(qvec)s) AS similarity
        FROM places
        WHERE embedding IS NOT NULL
    """
    params = {"qvec": query_embedding, "limit": limit}
    if category:
        sql += " AND category = %(category)s"
        params["category"] = category
    if neighborhood:
        sql += " AND neighborhood = %(neighborhood)s"
        params["neighborhood"] = neighborhood
    sql += " ORDER BY embedding <=> %(qvec)s LIMIT %(limit)s;"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


if __name__ == "__main__":
    db_url = os.environ["DATABASE_URL"]
    model = SentenceTransformer(MODEL_NAME)
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        register_vector(conn)

        # Same style of query as the earlier product mockup
        results = search("cozy quiet cafe good for working", conn, model, category="cafe")
        log.info("Query: 'cozy quiet cafe good for working' (category=cafe)")
        for r in results:
            log.info(f"  {r['name']} ({r['neighborhood']}) — similarity {r['similarity']:.3f}")
