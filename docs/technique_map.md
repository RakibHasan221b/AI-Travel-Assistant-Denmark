# Technique map

Updated as each phase lands — links point to the commit/file that first
implements the technique for real, not just an import.

| Technique | Phase | Genuine use case | Status |
|---|---|---|---|
| SQL | 1 | Schema, ingestion, aggregation views, agent's SQL tool | done — `db/schema.sql`, `ingestion/*.py` |
| Zero-shot prompting | 4 | Reddit post relevance classification (`pipeline/llm/prompts/relevance_zero_shot.py`) | tabled with Phase 4 |
| Few-shot prompting | 4, 7 | Aspect-sentiment labeling (`aspect_sentiment_few_shot.py`, tabled with Phase 4); cluster naming (`cluster_naming_few_shot.py`) | done (Phase 7 half) |
| ML→LLM→ML distillation loop | 5 | Cheap classifier replicates LLM relevance calls, LLM fallback only when uncertain | blocked on Phase 4 data |
| Supervised learning | 5, 9 | Relevance classification (blocked on Phase 4); quality-score regression | done (Phase 9 half) |
| Vectorization/embeddings | 5, 6 | TF-IDF for distillation; sentence-transformer embeddings for pgvector | done (Phase 6 half) — `pipeline/embeddings/embed_places.py` |
| pgvector semantic search | 6 | Semantic search + structured filtering in one SQL query, RAG retrieval backbone | done — `pipeline/embeddings/semantic_search.py`, verified with combined category+neighborhood+semantic queries |
| Unsupervised learning | 7 | Clustering into vibe collections | done — `pipeline/clustering/cluster_places.py`, 18 clusters, verified coherent (e.g. Sushi Restaurants cluster is 100% actual sushi places) |
| RAG | 8 | Grounded AI summaries with cited sources | done — `pipeline/rag/generate_summaries.py`, 175 places summarized, retrieval scoped per-place (never borrows another place's text), prompt explicitly flags thin evidence instead of writing confidently anyway |
| LangChain | 8 | RAG chain orchestration | done — `ChatPromptTemplate \| ChatOpenAI` chain in `pipeline/rag/generate_summaries.py` |
| XGBoost | 9 | Quality-score regression, tuned | done — winner, MSE 100.86, `pipeline/modeling/train_quality_model.py` |
| Random Forest | 9 (also 5) | Quality-score regression; distillation classifier option | done — MSE 104.11, lost to XGBoost narrowly |
| Neural networks | 9 | Small net in the quality-score bake-off | done — MLPRegressor, MSE 154.08, clearly lost on this data size (the intended "know when not to reach for DL" result) |
| Hyperparameter tuning | 9 | Optuna tuning XGBoost/Random Forest | done — 25 trials each, 3-fold CV |
| MSE / MAE | 9, 10 | Regression and forecast evaluation | done — Phase 9 (4-model bake-off) and Phase 10 (chronological forecast eval) |
| Time series | 10 | Weather-driven visit forecasting, chronological train/test split | done — `pipeline/timeseries/forecast_interest.py`, MSE 17.06/MAE 2.35, real Open-Meteo weather + confirmed Copenhagen event dates |
| CrewAI multi-agent crew | 11 | Trip-planning capstone — Place Scout, Conditions Analyst, and Concierge agents collaborate, each using crewai-native tools (SQL, pgvector, weather, quality-score) over Groq | done — `agent/crew.py`, `agent/tools.py`, `api/main.py`, verified live end-to-end via a real `POST /trip-plan` HTTP request |
| React / TypeScript / Node.js (optional) | 14 | Next.js frontend as an alternative client for the Phase 11 FastAPI service — deployed on Vercel; bolt-on, not required for the core pipeline | not started |
| Data validation | — | Pydantic gate on OSM ingestion rows (name, category, plausible coordinates, price range) before they reach `places` — catches malformed records the tag-classification step alone doesn't | done — `ingestion/validation.py`, wired into `osm_common.to_row()`, verified live via `osm_live_lookup.py`, covered by `tests/test_validation.py` |
| A/B testing | — | Two RAG-summary prompt variants (paragraph vs verdict-first) run on the same retrieved snippets, scored deterministically, logged to `experiment_results` | done — `pipeline/experiments/ab_test_rag_summaries.py`, run live on 10 places (20 real GPT-4o generations, $0.027): tally a=3, b=2, tie=5; scoring covered by `tests/test_ab_scoring.py` |
| Web-search enrichment | — | Backfills places with zero linked text (found live: the Little Mermaid statue ranked 701st/1,896 for its own name, only had a bare Danish OSM tag) via Serper.dev search, stored as ordinary `reviews_raw` rows so Phase 6/8 pick it up unchanged | done — `ingestion/web_enrichment.py`, verified live: Little Mermaid went from rank 701 (similarity 0.155) to rank 1 (similarity 0.66) after re-embedding, then got a real grounded AI summary; covered by `tests/test_web_enrichment.py` |
