"""Offline DistilBERT sentiment precompute — the fix for a real, measured
deployment problem, not a redesign.

Real memory test (this session) found loading DistilBERT's transformers
pipeline live, alongside the rest of the app, peaks at ~804 MB, well over
Render's 512 MB free-tier limit — the same class of OOM this project
already hit once before with sentence-transformers. The model design
doesn't change: DistilBERT still provides the sentiment signal, MiniLM
still provides the semantic signal, XGBoost still combines them. Only
*where* DistilBERT's inference happens moves, from live-per-request to
here, a one-time offline batch job, since Copenhagen's places are mostly
static and there's little value paying that memory cost on every request.

Scores one combined text per PLACE (all its reviews_raw text joined),
matching exactly what the live path already did before this change — not
per-review, since place_details() always combined a place's reviews into
one string before scoring. Stored in the existing ml_predictions table
(target='distilbert_sentiment'), the same real infrastructure
quality_score already used, not a new table — this project's own real
Explore/Stats data already lives there the same way.

Only computes for places missing a score, or whose review text has
genuinely changed since the last run (a simple review-count check, not a
full hash — cheap and good enough at this project's real data scale) —
re-running this doesn't waste recomputation on places already covered.
"""

import logging
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("precompute_distilbert_sentiment")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_places_needing_a_score(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.place_id,
                   array_agg(r.text_content ORDER BY r.review_id) AS reviews,
                   COUNT(r.review_id) AS review_count
            FROM places p
            JOIN reviews_raw r ON r.place_id = p.place_id
            LEFT JOIN ml_predictions m
                ON m.place_id = p.place_id AND m.target = 'distilbert_sentiment'
                AND m.model_name = %s
            GROUP BY p.place_id, m.confidence
            HAVING m.confidence IS NULL OR m.confidence != COUNT(r.review_id)
            """,
            (MODEL_NAME,),
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def save_score(conn, place_id, sentiment_score: float, review_count: int) -> None:
    with conn.cursor() as cur:
        # confidence doubles as "how many reviews this score was computed
        # from" — real, cheap staleness detection: if a place gets more
        # reviews later, review_count no longer matches and it gets
        # rescored on the next run, not silently left stale forever.
        cur.execute(
            "DELETE FROM ml_predictions WHERE place_id = %s AND target = 'distilbert_sentiment';",
            (place_id,),
        )
        cur.execute(
            """
            INSERT INTO ml_predictions (place_id, target, predicted_value, model_name, confidence)
            VALUES (%s, 'distilbert_sentiment', %s, %s, %s);
            """,
            (place_id, sentiment_score, MODEL_NAME, review_count),
        )
    conn.commit()


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        places = load_places_needing_a_score(conn)
        log.info(f"{len(places)} places need a DistilBERT sentiment score computed or refreshed")
        if not places:
            log.info("Nothing to do.")
            return

        from transformers import pipeline as hf_pipeline
        sentiment_clf = hf_pipeline("sentiment-analysis", model=MODEL_NAME)

        for i, place in enumerate(places):
            combined_text = " ".join(place["reviews"])[:512]
            result = sentiment_clf(combined_text)[0]
            score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
            save_score(conn, place["place_id"], score, place["review_count"])
            if (i + 1) % 20 == 0:
                log.info(f"  [{i+1}/{len(places)}] scored so far")

    log.info(f"Done. {len(places)} places scored and saved to ml_predictions.")


if __name__ == "__main__":
    main()
