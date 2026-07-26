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
| RAG | 8 | Grounded AI summaries with cited sources | not started |
| LangChain | 8 | RAG chain orchestration | not started |
| XGBoost | 9 | Quality-score regression, tuned | done — winner, MSE 100.86, `pipeline/modeling/train_quality_model.py` |
| Random Forest | 9 (also 5) | Quality-score regression; distillation classifier option | done — MSE 104.11, lost to XGBoost narrowly |
| Neural networks | 9 | Small net in the quality-score bake-off | done — MLPRegressor, MSE 154.08, clearly lost on this data size (the intended "know when not to reach for DL" result) |
| Hyperparameter tuning | 9 | Optuna tuning XGBoost/Random Forest | done — 25 trials each, 3-fold CV |
| MSE / MAE | 9, 10 | Regression and forecast evaluation | done (Phase 9 half) — all 4 models compared on held-out MSE/MAE |
| Time series | 10 | Weather-driven visit forecasting | not started |
| CrewAI multi-agent crew | 11 | Trip-planning capstone — Place Scout, Conditions Analyst, and Concierge agents collaborate, each using LangChain-wrapped tools (SQL, pgvector, weather, quality-score) | not started |
| React / TypeScript / Node.js (optional) | 14 | Next.js frontend as an alternative client for the Phase 11 FastAPI service — deployed on Vercel; bolt-on, not required for the core pipeline | not started |
