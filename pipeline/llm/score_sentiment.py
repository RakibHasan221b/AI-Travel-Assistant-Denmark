"""Aspect-sentiment scoring for already-linked review text.

Originally scoped as part of the Reddit pipeline (Phase 4), but the
technique doesn't actually require Reddit — it just needs real text linked
to a real place. Applied here to the 191 Wikivoyage descriptions already
linked to places (Phase 3), so Phase 9's quality-score model has real
sentiment data to build on without waiting on Reddit access.
"""

import logging
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "prompts"))
from groq_client import GroqError, call_groq  # noqa: E402
from aspect_sentiment_few_shot import ASPECT_SENTIMENT_PROMPT  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("score_sentiment")

VALID_ASPECTS = {"food", "service", "ambiance", "value", "location", "overall"}


def load_unscored_reviews(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.review_id, r.place_id, r.text_content
            FROM reviews_raw r
            WHERE r.place_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM aspect_sentiment a WHERE a.review_id = r.review_id
              );
            """
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def score_review(text: str) -> list[dict]:
    # Truncate very long descriptions - aspect sentiment doesn't need the whole essay
    prompt = ASPECT_SENTIMENT_PROMPT.format(text=text[:1500])
    result = call_groq(prompt)
    aspects = result.get("aspects", [])
    return [
        a for a in aspects
        if a.get("aspect") in VALID_ASPECTS and isinstance(a.get("sentiment_score"), int)
        and 1 <= a["sentiment_score"] <= 5
    ]


def refresh_aggregates(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM aggregated_sentiment;")
        cur.execute(
            """
            INSERT INTO aggregated_sentiment (place_id, aspect_category, avg_score, num_mentions, last_updated)
            SELECT place_id, aspect_category, AVG(sentiment_score), COUNT(*), now()
            FROM aspect_sentiment
            GROUP BY place_id, aspect_category;
            """
        )
    conn.commit()


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        reviews = load_unscored_reviews(conn)
        log.info(f"{len(reviews)} linked reviews to score")

        scored = failed = 0
        with conn.cursor() as cur:
            for i, review in enumerate(reviews):
                try:
                    aspects = score_review(review["text_content"])
                except GroqError as e:
                    log.warning(f"  [{i+1}/{len(reviews)}] failed, skipping: {e}")
                    failed += 1
                    continue

                for a in aspects:
                    cur.execute(
                        """
                        INSERT INTO aspect_sentiment (review_id, place_id, aspect_category,
                                                       sentiment_score, model_used)
                        VALUES (%s, %s, %s, %s, 'groq-llama-3.3-70b');
                        """,
                        (review["review_id"], review["place_id"], a["aspect"], a["sentiment_score"]),
                    )
                scored += 1
                if (i + 1) % 20 == 0:
                    conn.commit()
                    log.info(f"  [{i+1}/{len(reviews)}] scored so far")
                time.sleep(0.3)  # stay well under Groq's free-tier rate limit
        conn.commit()

        log.info(f"Scored {scored} reviews ({failed} failed), refreshing aggregates...")
        refresh_aggregates(conn)

    log.info("Done.")


if __name__ == "__main__":
    main()
