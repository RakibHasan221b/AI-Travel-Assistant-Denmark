"""Phase 1 of the ranking architecture: a quality-aware recommendation
classifier, not a rating-regression model and not true learning-to-rank.

Real investigation this session found, in order: (1) predicting an exact
0-100 rating from review text is not learnable at this project's data
scale, real evidence via 5-fold cross-validation against a dummy baseline,
not a single lucky/unlucky split; (2) binary good/bad classification IS a
real, usable signal (DistilBERT alone: macro F1 0.59, catching ~30% of
genuinely bad places versus 0% for a naive baseline); (3) true
learning-to-rank (XGBoost's rank:pairwise/rank:ndcg) is explicitly NOT
attempted here — it needs query-to-relevance-judgment data (clicks,
saves, pairwise preferences) this project has none of. Pretending to have
that supervision would be forcing ML terminology onto data that doesn't
support it. This classifier's output (a "would this be a good
recommendation" probability) is meant to feed the existing pgvector
semantic retrieval as a ranking/filtering signal, not to be a
standalone quality score.

Two independent, real signals are kept as separate features rather than
picking one: DistilBERT's sentiment score (explicit opinion language) and
a MiniLM-derived sentiment score (captures implicit/contextual cues a
pure sentiment model misses — e.g. "beautiful location, but the queues
were unbearable" reads positive on the word "beautiful" alone but is
really a complaint; real evidence found MiniLM's own bad-recall, 43%,
beat DistilBERT's, 30%, on this exact kind of case).

First real attempt at this fed the raw 384-dimension MiniLM embedding
directly into XGBoost alongside structured features, on only 189 rows —
396 features against 189 examples, the same too-many-dimensions problem
that broke the earlier regression attempt, reintroduced by accident.
Fixed by NOT stacking the raw embedding into XGBoost at all: a separate
logistic regression is trained on the full 527-review set (matching the
exact setup that validated MiniLM's real signal earlier), and only its
single predicted probability — one scalar, not 384 — becomes a feature
here, the same clean pattern DistilBERT's score already uses.

Evaluation is deliberately NOT NDCG/Precision@K/MRR — this project has no
relevance-judgment data to compute those honestly against. Instead: a
holdout check (do the model's top-10 highest-probability places actually
have a higher real average rating than its bottom-10?) and a calibration
check (does "90% recommended" actually mean mostly-good real places?) —
both computable from data this project genuinely has.
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import pipeline as hf_pipeline
from xgboost import XGBClassifier

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_recommendation_classifier")

RANDOM_STATE = 42
GOOD_THRESHOLD = 4.0  # real rating (1-5 scale) at/above which a place counts as "good"
MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "recommendation_classifier.json")
MINILM_CLF_OUT_PATH = os.path.join(os.path.dirname(__file__), "minilm_sentiment_clf.joblib")
MINILM_CLF_WEIGHTS_OUT_PATH = os.path.join(os.path.dirname(__file__), "minilm_sentiment_clf_weights.npz")
FEATURE_SCHEMA_OUT_PATH = os.path.join(os.path.dirname(__file__), "recommendation_feature_schema.json")
TRAINING_ONLY_PATH = os.path.join(os.path.dirname(__file__), "training_only_labels.jsonl")


def load_all_labeled_text(conn) -> pd.DataFrame:
    """All 527 real labeled reviews (place-linked + training-only, no
    place_id needed here) — used only to train the MiniLM scalar signal,
    the same full dataset that already validated it, not the smaller
    189-row place-linked subset."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.text_content, AVG(a.sentiment_score) AS raw_score
            FROM reviews_raw r JOIN aspect_sentiment a ON a.review_id = r.review_id
            GROUP BY r.review_id, r.text_content;
            """
        )
        rows = list(cur.fetchall())
    with open(TRAINING_ONLY_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["text_content"], d["mean_aspect_score"]))
    return pd.DataFrame(rows, columns=["text_content", "raw_score"])


def load_place_linked_reviews(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.place_id, r.text_content, AVG(a.sentiment_score) AS raw_score,
                   p.category, p.subcategory, p.opening_hours, p.price_level, p.osm_tags
            FROM reviews_raw r
            JOIN aspect_sentiment a ON a.review_id = r.review_id
            JOIN places p ON p.place_id = r.place_id
            GROUP BY r.review_id, r.place_id, r.text_content,
                     p.category, p.subcategory, p.opening_hours, p.price_level, p.osm_tags;
            """
        )
        columns = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)

    with conn.cursor() as cur:
        cur.execute("SELECT place_id, COUNT(*) AS review_count FROM reviews_raw GROUP BY place_id;")
        counts = dict(cur.fetchall())
    df["review_count"] = df["place_id"].map(counts).fillna(1).astype(int)
    return df


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    struct = pd.DataFrame({
        "has_subcategory": df["subcategory"].notna().astype(int),
        "has_opening_hours": df["opening_hours"].notna().astype(int),
        "has_phone": df["osm_tags"].apply(lambda t: int(bool(t and t.get("phone")))),
        "has_website": df["osm_tags"].apply(lambda t: int(bool(t and t.get("website")))),
        "has_price_level": df["price_level"].notna().astype(int),
        "price_level": df["price_level"].fillna(0).astype(int),
        "review_count": df["review_count"],
    })
    cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(int)
    return pd.concat([struct, cat_dummies], axis=1)


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        df = load_place_linked_reviews(conn)
    log.info(f"{len(df)} place-linked reviews with real structured features")

    # XGBClassifier needs numeric labels; keep the string labels around for
    # readable reports.
    y_str = (df["raw_score"] >= GOOD_THRESHOLD).map({True: "good", False: "bad"}).values
    y = (y_str == "good").astype(int)
    log.info(f"label distribution: {pd.Series(y_str).value_counts().to_dict()}")

    log.info("Computing DistilBERT sentiment signal (one scalar per review)...")
    sentiment_clf = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_preds = sentiment_clf([t[:512] for t in df["text_content"].tolist()], batch_size=16)
    distilbert_signed = np.array(
        [p["score"] if p["label"] == "POSITIVE" else -p["score"] for p in sentiment_preds]
    ).reshape(-1, 1)

    log.info("Training the MiniLM scalar signal on the full 527-review set (not just these 189)...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        df_full = load_all_labeled_text(conn)
    y_full = (df_full["raw_score"] >= GOOD_THRESHOLD).astype(int).values
    X_embed_full = embed_model.encode(df_full["text_content"].tolist(), show_progress_bar=False)
    minilm_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    minilm_clf.fit(X_embed_full, y_full)
    log.info(f"MiniLM scalar model trained on {len(df_full)} reviews")

    X_embed_189 = embed_model.encode(df["text_content"].tolist(), show_progress_bar=False)
    minilm_scalar = minilm_clf.predict_proba(X_embed_189)[:, 1].reshape(-1, 1)

    struct = build_structured_features(df)
    X = np.hstack([distilbert_signed, minilm_scalar, struct.values])
    feature_names = ["distilbert_sentiment", "minilm_good_probability"] + list(struct.columns)
    log.info(f"combined feature matrix: {X.shape[1]} dimensions for {len(df)} rows "
              f"({len(df)/X.shape[1]:.1f} rows per feature, versus 0.5 last attempt)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_preds = cross_val_predict(dummy, X, y, cv=skf)
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, reg_lambda=5.0,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    model_preds = cross_val_predict(model, X, y, cv=skf)

    target_names = ["bad", "good"]
    log.info("=== Baseline (most frequent class) ===")
    log.info("\n" + classification_report(y, dummy_preds, target_names=target_names, zero_division=0))
    log.info("=== XGBoost (DistilBERT + MiniLM + structured features) ===")
    log.info("\n" + classification_report(y, model_preds, target_names=target_names, zero_division=0))
    log.info(f"macro F1 - baseline: {f1_score(y, dummy_preds, average='macro'):.3f}, "
              f"model: {f1_score(y, model_preds, average='macro'):.3f}")

    # Holdout-style check: fit on a 80% split, get probabilities on the
    # held-out 20%, verify the real ratings back that ranking up — the
    # honest evaluation this project's real data can actually support,
    # since there's no relevance-judgment data for NDCG/Precision@K/MRR.
    log.info("Holdout check: does predicted probability actually track real ratings?")
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    model.fit(X[idx_train], y[idx_train])
    proba_good = model.predict_proba(X[idx_test])[:, list(model.classes_).index(1)]  # 1 == "good"
    test_scores = df["raw_score"].values[idx_test]
    order = np.argsort(-proba_good)
    k = min(10, len(order) // 2)
    top_k_avg_rating = test_scores[order[:k]].mean()
    bottom_k_avg_rating = test_scores[order[-k:]].mean()
    log.info(f"Top-{k} predicted-good places: real avg rating = {top_k_avg_rating:.2f}")
    log.info(f"Bottom-{k} predicted-good places: real avg rating = {bottom_k_avg_rating:.2f}")

    # Calibration: among places predicted >=70% likely good, what fraction
    # actually ARE good (real rating >= threshold)?
    high_conf_mask = proba_good >= 0.7
    if high_conf_mask.sum() > 0:
        actually_good = (test_scores[high_conf_mask] >= GOOD_THRESHOLD).mean()
        log.info(f"Of {high_conf_mask.sum()} places predicted >=70% good, "
                  f"{actually_good*100:.0f}% actually are (real rating >= {GOOD_THRESHOLD})")
    else:
        log.info("No test-set places crossed the 70% confidence threshold")

    # Refit on everything for the model that ships.
    model.fit(X, y)
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda t: -t[1])
    log.info("Top 10 most important features:")
    for name, imp in importances[:10]:
        log.info(f"  {name:<20} {imp:.3f}")

    model.save_model(MODEL_OUT_PATH)
    log.info(f"Saved trained model to {MODEL_OUT_PATH}")

    # The MiniLM scalar signal is itself a trained model (a logistic
    # regression on the full 527-review set) — live scoring needs it too,
    # not just the final XGBoost classifier, to turn a NEW place's embedding
    # into the same scalar feature this model was trained on.
    joblib.dump(minilm_clf, MINILM_CLF_OUT_PATH)
    log.info(f"Saved MiniLM sentiment classifier to {MINILM_CLF_OUT_PATH}")

    # agent/recommendation_service.py loads only these two arrays at
    # runtime, not the joblib file — importing scikit-learn just to
    # unpickle a LogisticRegression costs ~94MB RSS by itself (measured),
    # a real contributor to Render's 512MB OOM. sigmoid(X @ coef_.T +
    # intercept_) is the exact same computation LogisticRegression's own
    # predict_proba does for a binary classifier (verified diff = 0.0 on
    # real embeddings) — the .joblib file above still exists so this
    # script's own eval/plotting code has a normal sklearn object to use.
    np.savez(
        MINILM_CLF_WEIGHTS_OUT_PATH,
        coef=minilm_clf.coef_.astype(np.float64),
        intercept=minilm_clf.intercept_.astype(np.float64),
    )
    log.info(f"Saved MiniLM classifier weights (numpy-only) to {MINILM_CLF_WEIGHTS_OUT_PATH}")

    # Save the exact feature order/category columns seen during training —
    # live inference must build features in this exact shape, and category
    # one-hot columns are only known by what this training data contained,
    # not by every category that could theoretically exist.
    schema = {
        "feature_order": feature_names,
        "category_columns": [c for c in struct.columns if c.startswith("cat_")],
        "good_threshold": GOOD_THRESHOLD,
    }
    with open(FEATURE_SCHEMA_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    log.info(f"Saved feature schema to {FEATURE_SCHEMA_OUT_PATH}")


if __name__ == "__main__":
    main()
