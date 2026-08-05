"""Real-time Rating model: embed raw review text, predict one 0-100 score
directly, no LLM anywhere in the live path.

Replaces the LLM-based sentiment scoring pipeline (pipeline/llm/
score_sentiment.py) for live use. The existing aspect_sentiment table
(real per-review, per-aspect scores the LLM produced in the past) is
reused exactly once, offline, here, as the labeled training set — the LLM
already did its part; this script never calls one. Each review's aspect
scores (food/service/ambiance/value/location/overall) get averaged into
one label per review, its text embedded with sentence-transformers, and
XGBoost learns to predict that single number directly from the embedding.
At live inference time, the same embedding model plus this trained model
replace every LLM call sentiment/quality scoring used to make — see
pipeline/modeling/rating.py for the live scoring function this trains for.

Two label sources get combined: aspect_sentiment (place-linked, usable
live) plus training_only_labels.jsonl (reviews with no place_id, so
unusable for the live app's per-place sentiment display, but their real
text and real scores are still valid training signal — see
pipeline/llm/score_sentiment_training_only.py for how that file was
built). Combined dataset was 189 + 338 = 527 real labeled reviews at last
count, versus 188 originally — real evidence from a held-out comparison
showed the original set was too small and too skewed toward high scores
for any model to beat a dummy mean-baseline; more, more balanced data was
the actual fix, not a different model.

A genuine held-out test split is used and RMSE/MAE/R²/success-rate
reported on it, compared against a dummy baseline, before this model is
trusted live, same rigor as train_quality_model.py's own bake-off.
"""

import json
import logging
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_rating_model")

RANDOM_STATE = 42
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "rating_model.json")
TRAINING_ONLY_PATH = os.path.join(os.path.dirname(__file__), "training_only_labels.jsonl")

# Same tolerance and same reason train_quality_model.py uses it: a plain-
# English number for anyone who'd rather skip RMSE/R², and MAPE isn't used
# because these are engineered scores that can legitimately sit near zero.
SUCCESS_TOLERANCE = 10

MIN_LABELED_REVIEWS = 20


def load_db_labeled_reviews(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.review_id, r.text_content, AVG(a.sentiment_score) AS mean_aspect_score
            FROM reviews_raw r
            JOIN aspect_sentiment a ON a.review_id = r.review_id
            GROUP BY r.review_id, r.text_content;
            """
        )
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def load_training_only_reviews() -> pd.DataFrame:
    if not os.path.exists(TRAINING_ONLY_PATH):
        return pd.DataFrame(columns=["review_id", "text_content", "mean_aspect_score"])
    rows = []
    with open(TRAINING_ONLY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append({
                    "review_id": d["review_id"],
                    "text_content": d["text_content"],
                    "mean_aspect_score": d["mean_aspect_score"],
                })
    return pd.DataFrame(rows)


def build_target(mean_aspect_score: pd.Series) -> pd.Series:
    # 1-5 aspect scale rescaled to 0-100 — the same rescale
    # train_quality_model.py already uses for its own sentiment component,
    # so every score in this app sits on one consistent 0-100 scale.
    return (mean_aspect_score.astype(float) - 1) / 4 * 100


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        df_db = load_db_labeled_reviews(conn)
    df_training_only = load_training_only_reviews()
    df = pd.concat([df_db, df_training_only], ignore_index=True)

    log.info(
        f"{len(df_db)} place-linked (aspect_sentiment) + {len(df_training_only)} "
        f"training-only (no place_id) = {len(df)} total labeled reviews"
    )
    if len(df) < MIN_LABELED_REVIEWS:
        raise SystemExit(
            f"Only {len(df)} labeled reviews — too few to train and honestly "
            f"validate a model (need at least {MIN_LABELED_REVIEWS})."
        )

    y = build_target(df["mean_aspect_score"])
    log.info(f"Label distribution: min={y.min():.1f} max={y.max():.1f} mean={y.mean():.1f} std={y.std():.1f}")

    log.info(f"Embedding {len(df)} reviews with {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    X = embed_model.encode(df["text_content"].tolist(), show_progress_bar=False)

    # Stratified by score bucket, not a plain random split — with a modest,
    # real dataset like this, an unlucky random split can make train and
    # test look like different distributions by chance, which showed up as
    # a real, misleading gap during earlier testing on the smaller dataset.
    strata = pd.cut(y, bins=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strata
    )
    log.info(f"Train: {len(X_train)}, test (held out, never trained on): {len(X_test)}")

    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    for name, m in [("dummy_mean_baseline", dummy), ("xgboost", model)]:
        preds = m.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        within_tolerance = (abs(preds - y_test) <= SUCCESS_TOLERANCE).mean() * 100
        log.info(
            f"{name:<22} RMSE={rmse:6.2f}  MAE={mae:6.2f}  R2={r2:7.3f}  "
            f"within +/-{SUCCESS_TOLERANCE}pts={within_tolerance:4.0f}%"
        )

    # Refit on ALL labeled data (train + test) for the model that actually
    # ships — the held-out split above exists purely to report honest
    # numbers, same pattern train_quality_model.py's final_model uses.
    model.fit(X, y)
    model.save_model(MODEL_OUT_PATH)
    log.info(f"Saved trained model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
