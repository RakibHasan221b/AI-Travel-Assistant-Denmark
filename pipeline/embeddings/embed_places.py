"""Phase 6 — generate sentence embeddings for places and raw review/description
text, store them in the pgvector columns already declared in the schema.

Local model (sentence-transformers all-MiniLM-L6-v2, 384 dims, matches
`vector(384)` in schema.sql) — no API cost, no rate limits, runs on CPU.

Two things get embedded:
- places: a combined text blob (name + category + neighborhood + any linked
  review/Wikivoyage text) — powers semantic place search.
- reviews_raw: each row's own text_content — powers RAG retrieval in Phase 8.
"""

import logging
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("embed_places")

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def build_place_text(row: dict) -> str:
    parts = [row["name"], row["category"]]
    if row["subcategory"]:
        parts.append(row["subcategory"])
    if row["neighborhood"]:
        parts.append(f"in {row['neighborhood']}")
    if row["linked_text"]:
        parts.append(row["linked_text"])
    return ". ".join(parts)


def embed_places(conn, model: SentenceTransformer):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.place_id, p.name, p.category, p.subcategory, p.neighborhood,
                   string_agg(DISTINCT r.text_content, ' ') AS linked_text
            FROM places p
            LEFT JOIN place_mentions pm ON pm.place_id = p.place_id AND pm.confidence >= 0.5
            LEFT JOIN reviews_raw r ON r.review_id = pm.review_id
            GROUP BY p.place_id, p.name, p.category, p.subcategory, p.neighborhood;
            """
        )
        columns = [d.name for d in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    log.info(f"Embedding {len(rows)} places...")
    texts = [build_place_text(r) for r in rows]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

    with conn.cursor() as cur:
        for row, emb in zip(rows, embeddings):
            cur.execute(
                "UPDATE places SET embedding = %s, updated_at = now() WHERE place_id = %s;",
                (emb, row["place_id"]),
            )
    conn.commit()
    log.info(f"Done embedding {len(rows)} places")


def embed_reviews(conn, model: SentenceTransformer):
    with conn.cursor() as cur:
        cur.execute("SELECT review_id, text_content FROM reviews_raw WHERE embedding IS NULL;")
        rows = cur.fetchall()

    log.info(f"Embedding {len(rows)} reviews_raw rows...")
    texts = [r[1] for r in rows]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

    with conn.cursor() as cur:
        for (review_id, _), emb in zip(rows, embeddings):
            cur.execute(
                "UPDATE reviews_raw SET embedding = %s WHERE review_id = %s;", (emb, review_id)
            )
    conn.commit()
    log.info(f"Done embedding {len(rows)} reviews")


def main():
    db_url = os.environ["DATABASE_URL"]
    log.info(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        register_vector(conn)
        embed_places(conn, model)
        embed_reviews(conn, model)


if __name__ == "__main__":
    main()
