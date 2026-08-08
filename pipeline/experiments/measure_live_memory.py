"""Real, staged process-memory measurement of the deployed API entrypoint
(api.main, the exact module `uvicorn api.main:app` loads on Render) — not
an estimate. Run standalone: `python pipeline/experiments/measure_live_memory.py`.

Real finding this validated (2026-08-05): relocating DistilBERT sentiment
to an offline batch job (pipeline/modeling/precompute_distilbert_sentiment.py)
brought the live peak from ~804MB down to ~410MB cold, ~436MB under
concurrent load. That earlier number was itself incomplete — it only
exercised api.main's import + predict_recommendation(), never a real
CrewAI Crew/Agent/Task construction or kickoff(). A real Render OOM email
(2026-08-08) confirmed the process does exceed 512MB, which this
narrower test never reproduced. Root-caused since: `import sklearn` alone
costs ~94MB RSS, and the combined stack (FastEmbed + MiniLM/XGBoost +
one build_crew()) landed at ~522MB even before a single real request —
over budget on its own. Fixed by dropping scikit-learn from the `agent`
extra entirely (agent/recommendation_service.py's MiniLM classifier now
computes its own numpy sigmoid instead of unpickling a sklearn object,
and the XGBoost model loads via the native Booster/DMatrix API instead
of the sklearn-requiring XGBClassifier wrapper — see that module's
docstring). This script now also builds a real Crew (no kickoff(), so no
Groq cost) to include that construction cost, which the original version
of this script omitted.

Re-run this after any change touching agent/tools.py, agent/crew.py,
agent/recommendation_service.py, or api/main.py's import graph — a
passing run before doesn't guarantee one after.

Measured on Windows; Render's containers run Linux, so treat this as a
strong pre-deployment signal, not a Linux-exact number — cross-check
against Render's own memory graph after deploying.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

process = psutil.Process(os.getpid())


def rss_mb() -> float:
    return process.memory_info().rss / (1024 * 1024)


def _test_place(i: int, category: str = "restaurant") -> dict:
    return {
        "name": f"Test Place {i}",
        "category": category,
        "subcategory": "danish",
        "opening_hours": "Mon-Sun 10:00-22:00",
        "price_level": 2,
        "has_phone": True,
        "has_website": True,
        "review_count": 2,
        "reviews": [f"Great food, visit {i}.", "Nice place, would come back."],
        "distilbert_sentiment_score": 0.9,
    }


def main():
    print(f"cold start                                            {rss_mb():.0f} MB")

    t0 = time.time()
    import api.main as api_main
    print(f"+ module import (cold startup, like Render boot)     {rss_mb():.0f} MB  ({time.time()-t0:.1f}s)")

    from agent.recommendation_service import (
        _get_minilm_clf_weights,
        _get_xgb_model,
        predict_recommendation,
    )

    t0 = time.time()
    model = api_main.get_embed_model()
    next(model.embed(["warm-up call to force real weight loading"]))
    predict_recommendation(_test_place(0))
    print(f"+ first real request (cold-path model loads)          {rss_mb():.0f} MB  ({time.time()-t0:.1f}s)")

    embed_a, embed_b = api_main.get_embed_model(), api_main.get_embed_model()
    xgb_a, xgb_b = _get_xgb_model(), _get_xgb_model()
    minilm_a, minilm_b = _get_minilm_clf_weights(), _get_minilm_clf_weights()
    assert embed_a is embed_b, "fastembed model was NOT a singleton — reloaded!"
    assert xgb_a is xgb_b, "xgboost model was NOT a singleton — reloaded!"
    assert minilm_a is minilm_b, "minilm classifier weights were NOT a singleton — reloaded!"
    print("singleton check: fastembed, xgboost, minilm classifier weights each confirmed a single shared instance")

    t0 = time.time()
    for i in range(30):
        predict_recommendation(_test_place(i, category="cafe"))
    print(f"+ after 30 sequential requests                        {rss_mb():.0f} MB  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda i: predict_recommendation(_test_place(i, "hotel")), range(40)))
    print(f"+ after 40 concurrent requests (8 threads)             {rss_mb():.0f} MB  ({time.time()-t0:.1f}s)   <- peak")
    print(f"  all 40 concurrent results well-formed: {all('recommendation_probability' in r for r in results)}")

    t0 = time.time()
    from agent.crew import build_crew
    crew = build_crew()
    del crew
    print(f"+ one build_crew() (Agent/Task/Crew/LLM construction, no kickoff, no Groq cost)   {rss_mb():.0f} MB  ({time.time()-t0:.1f}s)   <- ready-to-serve baseline")


if __name__ == "__main__":
    main()
