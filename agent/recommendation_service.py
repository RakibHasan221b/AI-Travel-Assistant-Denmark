"""Live recommendation-probability scoring — the single source of truth
for "would this place likely be a good recommendation," computed fresh
from a place's CURRENT review signals and real metadata, never a stored
final score. This is the real inference wrapper around the trained Phase 1
classifier (pipeline/modeling/train_recommendation_classifier.py) — see
that script's docstring and the accompanying investigation report for why
this predicts a recommendation probability, not an exact 0-100 quality
score, and why DistilBERT and MiniLM are kept as two separate signals.

MiniLM reuses agent/tools.py's existing shared fastembed instance
(get_embed_model()), not a second sentence-transformers copy — that
module's own docstring documents why: sentence-transformers pulls in
torch, which already caused a real Render free-tier (512MB) OOM crash
once. fastembed's ONNX export of the same all-MiniLM-L6-v2 weights was
already verified (cosine similarity 1.0000) to produce the same vectors,
so it's a safe drop-in for this trained classifier's expected input too.

DistilBERT sentiment is NOT computed live here. A real memory measurement
(psutil, staged process) found loading transformers' DistilBERT pipeline
in the live process peaks at ~804MB, well over Render's 512MB free-tier
limit — the same OOM class already hit once with sentence-transformers.
Since Copenhagen's places are mostly static, DistilBERT sentiment is
precomputed once, offline (pipeline/modeling/precompute_distilbert_sentiment.py)
and stored in ml_predictions (target='distilbert_sentiment'). Callers of
predict_recommendation() must fetch that stored value themselves (agent/
tools.py does this — a plain SELECT, no torch import in this process at
all) and pass it in as place["distilbert_sentiment_score"]. The model
architecture itself is unchanged — DistilBERT still provides this exact
signal — only where its inference runs moved, from live to offline.

Models load once, lazily, on first use, and stay in memory — the same
pattern agent/tools.py already uses, not reloaded per request.
"""

import json
import os

import numpy as np
from xgboost import XGBClassifier

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "pipeline", "modeling")
_XGB_PATH = os.path.join(_MODELS_DIR, "recommendation_classifier.json")
_MINILM_CLF_PATH = os.path.join(_MODELS_DIR, "minilm_sentiment_clf.joblib")
_SCHEMA_PATH = os.path.join(_MODELS_DIR, "recommendation_feature_schema.json")

_xgb_model: XGBClassifier | None = None
_minilm_clf = None
_schema: dict | None = None


def _get_xgb_model() -> XGBClassifier:
    global _xgb_model
    if _xgb_model is None:
        _xgb_model = XGBClassifier()
        _xgb_model.load_model(_XGB_PATH)
    return _xgb_model


def _get_minilm_clf():
    global _minilm_clf
    if _minilm_clf is None:
        import joblib
        _minilm_clf = joblib.load(_MINILM_CLF_PATH)
    return _minilm_clf


def _get_schema() -> dict:
    global _schema
    if _schema is None:
        with open(_SCHEMA_PATH, encoding="utf-8") as f:
            _schema = json.load(f)
    return _schema


def _minilm_good_probability(text: str) -> float:
    """Probability in [0, 1] that a MiniLM-embedded review reads as
    'good' — from agent/tools.py's shared fastembed instance, the same
    embedding space the training script used sentence-transformers for."""
    from agent.tools import get_embed_model

    embedding = next(get_embed_model().embed([text]))
    proba = _get_minilm_clf().predict_proba(np.array(embedding).reshape(1, -1))[0]
    classes = list(_get_minilm_clf().classes_)
    return float(proba[classes.index(1)])


def extract_features(place: dict) -> tuple[np.ndarray, dict]:
    """place is a dict with real fields matching what the training data
    had: name, reviews (list[str], real raw review text — may be empty),
    category, subcategory, opening_hours, price_level, has_phone,
    has_website, review_count, distilbert_sentiment_score (precomputed
    offline — see this module's docstring — or None if not yet scored).
    Returns (feature_vector, signals_dict) — the vector for the model,
    the signals dict for an honest, explainable response.

    Deliberately does NOT take rating as input — rating is what the
    training label was derived from, using it as a feature would be
    leaking the answer into the question."""
    schema = _get_schema()
    reviews = place.get("reviews") or []
    combined_text = " ".join(reviews) if reviews else ""
    has_distilbert_score = place.get("distilbert_sentiment_score") is not None

    distilbert_score = float(place["distilbert_sentiment_score"]) if has_distilbert_score else 0.0
    if combined_text.strip():
        minilm_score = _minilm_good_probability(combined_text)
    else:
        # No review text at all — real, honest null-signal case, not a
        # guess. Callers should check signals["has_review_text"] before
        # trusting the probability with full confidence.
        minilm_score = 0.5

    has_subcategory = int(bool(place.get("subcategory")))
    has_opening_hours = int(bool(place.get("opening_hours")))
    has_phone = int(bool(place.get("has_phone")))
    has_website = int(bool(place.get("has_website")))
    price_level = place.get("price_level")
    has_price_level = int(price_level is not None)
    review_count = int(place.get("review_count") or 0)

    struct_by_name = {
        "has_subcategory": has_subcategory,
        "has_opening_hours": has_opening_hours,
        "has_phone": has_phone,
        "has_website": has_website,
        "has_price_level": has_price_level,
        "price_level": price_level or 0,
        "review_count": review_count,
    }
    category = place.get("category", "")
    for col in schema["category_columns"]:
        struct_by_name[col] = int(col == f"cat_{category}")

    feature_by_name = {"distilbert_sentiment": distilbert_score, "minilm_good_probability": minilm_score}
    feature_by_name.update(struct_by_name)

    vector = np.array([feature_by_name[name] for name in schema["feature_order"]], dtype=float).reshape(1, -1)
    signals = {
        "sentiment_distilbert": round(distilbert_score, 3),
        "sentiment_minilm": round(minilm_score, 3),
        "has_review_text": bool(combined_text.strip()),
        "has_distilbert_score": has_distilbert_score,
    }
    return vector, signals


def predict_recommendation(place: dict) -> dict:
    """The real, live scoring entry point. Returns a plain dict, not a
    Pydantic model — this is the "single source of truth" function other
    code (place_details, an API route) wraps as needed."""
    vector, signals = extract_features(place)
    model = _get_xgb_model()
    probability = float(model.predict_proba(vector)[0][1])  # class 1 == "good"
    return {
        "recommendation_probability": round(probability, 3),
        "label": "recommended" if probability >= 0.5 else "not recommended",
        "signals": signals,
    }
