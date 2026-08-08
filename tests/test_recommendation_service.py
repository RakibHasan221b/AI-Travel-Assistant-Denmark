"""Real regression coverage for agent/recommendation_service.py — this
module had zero test coverage before, which is how the sklearn-removal
change (see pyproject.toml's `agent` extra comment) went unverified by
any automated check. A real, staged memory measurement found `import
sklearn` alone costs ~94MB RSS and was a major contributor to a confirmed
Render production OOM; these tests guard against that dependency quietly
creeping back in, and confirm the numpy/native-Booster replacements are
numerically wired correctly — deterministic, no live DistilBERT
inference, no Groq/network call required at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agent.recommendation_service import (
    _get_minilm_clf_weights,
    _get_xgb_model,
    _minilm_good_probability,
    predict_recommendation,
)

_REAL_PLACE = {
    "name": "Restaurant Klubben",
    "category": "restaurant",
    "subcategory": "danish",
    "opening_hours": "11:00-23:00",
    "price_level": 2,
    "has_phone": True,
    "has_website": True,
    "review_count": 1,
    "reviews": ["Cozy and traditional Danish ambiance, good value for the price."],
    "distilbert_sentiment_score": 0.997,
}


def test_minilm_weights_are_plain_numpy_not_an_unpickled_sklearn_object():
    # The real fix this guards: the old path unpickled a joblib-saved
    # sklearn LogisticRegression, which needs scikit-learn importable just
    # to load the file. Regressing to that would only show up here — a
    # dev venv with the `modeling`/`distillation` extras installed already
    # has scikit-learn present (for offline training), and xgboost's own
    # compat shim opportunistically imports it whenever it's present
    # regardless of what this module does — so "sklearn not in
    # sys.modules" isn't a meaningful check in that shared venv. What IS
    # meaningful, and works everywhere: the loaded object is a plain numpy
    # array pair, not an sklearn estimator, which is what actually makes
    # scikit-learn droppable from Render's real `agent`-only install
    # (verified separately, in a venv with scikit-learn genuinely absent).
    coef, intercept = _get_minilm_clf_weights()
    assert isinstance(coef, np.ndarray)
    assert isinstance(intercept, np.ndarray)
    assert coef.shape == (1, 384)


def test_predict_recommendation_is_deterministic():
    first = predict_recommendation(_REAL_PLACE)
    second = predict_recommendation(_REAL_PLACE)
    assert first == second


def test_predict_recommendation_returns_well_formed_result():
    result = predict_recommendation(_REAL_PLACE)
    assert 0.0 <= result["recommendation_probability"] <= 1.0
    assert result["label"] in ("recommended", "not recommended")
    assert result["signals"]["has_review_text"] is True
    assert result["signals"]["has_distilbert_score"] is True


def test_predict_recommendation_does_not_require_distilbert_score():
    # DistilBERT sentiment is precomputed offline and passed in — a place
    # that was never scored (has_distilbert_score=False) must still get a
    # real recommendation, not crash, with the gap reflected honestly in
    # signals rather than guessed.
    place = dict(_REAL_PLACE, distilbert_sentiment_score=None)
    result = predict_recommendation(place)
    assert result["signals"]["has_distilbert_score"] is False


def test_predict_recommendation_handles_no_review_text_as_a_neutral_signal():
    place = dict(_REAL_PLACE, reviews=[])
    result = predict_recommendation(place)
    assert result["signals"]["has_review_text"] is False
    assert result["signals"]["sentiment_minilm"] == 0.5


def test_minilm_probability_matches_manual_sigmoid_of_exported_weights():
    # Confirms the wiring, not just that a number came out: recomputes the
    # expected probability directly from the exported .npz weights and a
    # real FastEmbed embedding, independent of _minilm_good_probability's
    # own internals, and checks they agree exactly.
    text = "Great food, wonderful atmosphere, highly recommend visiting."
    from agent.tools import get_embed_model

    embedding = np.array(next(get_embed_model().embed([text]))).reshape(1, -1)
    coef, intercept = _get_minilm_clf_weights()
    z = embedding @ coef.T + intercept
    expected = float(1 / (1 + np.exp(-z[0][0])))

    actual = _minilm_good_probability(text)
    assert actual == expected


def test_xgb_model_loads_via_native_booster_not_sklearn_wrapper():
    import xgboost as xgb

    model = _get_xgb_model()
    assert isinstance(model, xgb.Booster)


def test_xgb_model_is_a_singleton_not_reloaded_per_call():
    assert _get_xgb_model() is _get_xgb_model()


def test_agent_extra_does_not_declare_scikit_learn_or_joblib():
    # The actual, environment-independent guarantee: Render only ever
    # installs `pip install -e ".[agent]"`, so what matters is what THIS
    # extra declares, not whether scikit-learn happens to be importable in
    # whatever venv the tests run in (a shared dev venv with `modeling`/
    # `distillation` installed will have it regardless — see the sibling
    # tests above for why "sklearn not in sys.modules" isn't the right
    # check). scikit-learn/joblib should only ever appear under `modeling`
    # (offline training).
    import tomllib

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)
    agent_extra = config["project"]["optional-dependencies"]["agent"]
    assert "scikit-learn" not in agent_extra
    assert "joblib" not in agent_extra
