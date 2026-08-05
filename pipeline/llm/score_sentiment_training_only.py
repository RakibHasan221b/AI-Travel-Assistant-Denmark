"""One-time, training-only expansion of the Rating model's labeled dataset.

Scores reviews_raw rows that have NO place_id (never matched to a curated
place during ingestion, so they're unusable for the live app's per-place
sentiment feature) using the exact same real prompt/model as
score_sentiment.py. aspect_sentiment.place_id is NOT NULL in the schema,
so these can never be inserted there — instead, results are saved to a
separate local file, used only to train pipeline/modeling/
train_rating_model.py, never touching the live app or its database.

Resumable across sessions: already-scored review_ids are read back out of
OUT_PATH on startup and excluded from the DB fetch, and new results are
appended, never overwriting what a previous run already saved. Real Groq
free-tier daily/per-minute limits mean one run rarely finishes 362 reviews
in one sitting — re-running this script picks up exactly where the last
one left off instead of re-scoring (and re-spending tokens on) reviews
already done, or erasing them.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "prompts"))
from aspect_sentiment_few_shot import ASPECT_SENTIMENT_PROMPT
from groq_client import GroqError, call_groq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("score_sentiment_training_only")

VALID_ASPECTS = {"food", "service", "ambiance", "value", "location", "overall"}
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "modeling", "training_only_labels.jsonl")


def load_already_scored_review_ids() -> set[str]:
    if not os.path.exists(OUT_PATH):
        return set()
    ids = set()
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["review_id"])
    return ids


def load_unlinked_unscored_reviews(conn, already_scored: set[str]) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.review_id, r.text_content
            FROM reviews_raw r
            WHERE r.place_id IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM aspect_sentiment a WHERE a.review_id = r.review_id
              );
            """
        )
        columns = [d.name for d in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return [r for r in rows if str(r["review_id"]) not in already_scored]


def score_review(text: str) -> list[dict]:
    prompt = ASPECT_SENTIMENT_PROMPT.format(text=text[:1500])
    result = call_groq(prompt)
    aspects = result.get("aspects", [])
    return [
        a for a in aspects
        if a.get("aspect") in VALID_ASPECTS and isinstance(a.get("sentiment_score"), int)
        and 1 <= a["sentiment_score"] <= 5
    ]


def main():
    already_scored = load_already_scored_review_ids()
    log.info(f"{len(already_scored)} already scored in a previous run, resuming from there")

    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        reviews = load_unlinked_unscored_reviews(conn, already_scored)
    log.info(f"{len(reviews)} unlinked reviews left to score, training-only")

    scored = failed = empty = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, review in enumerate(reviews):
            try:
                aspects = score_review(review["text_content"])
            except GroqError as e:
                log.warning(f"  [{i+1}/{len(reviews)}] failed, skipping: {e}")
                failed += 1
                continue

            if not aspects:
                empty += 1
                time.sleep(0.3)
                continue

            mean_score = sum(a["sentiment_score"] for a in aspects) / len(aspects)
            out.write(json.dumps({
                "review_id": str(review["review_id"]),
                "text_content": review["text_content"],
                "aspects": aspects,
                "mean_aspect_score": mean_score,
            }) + "\n")
            scored += 1
            if (i + 1) % 20 == 0:
                out.flush()
                log.info(f"  [{i+1}/{len(reviews)}] scored so far ({scored} kept, {empty} empty, {failed} failed)")
            time.sleep(0.3)  # stay well under Groq's free-tier rate limit

    log.info(f"Done. {scored} scored and saved, {empty} had no real opinion expressed, {failed} failed.")
    log.info(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
