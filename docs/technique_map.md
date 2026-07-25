# Technique map

Updated as each phase lands — links point to the commit/file that first
implements the technique for real, not just an import.

| Technique | Phase | Genuine use case | Status |
|---|---|---|---|
| SQL | 1 | Schema, ingestion, aggregation views, agent's SQL tool | not started |
| Zero-shot prompting | 4 | Reddit post relevance classification (`pipeline/llm/prompts/relevance_zero_shot.py`) | not started |
| Few-shot prompting | 4, 7 | Aspect-sentiment labeling (`aspect_sentiment_few_shot.py`); cluster naming (`cluster_naming_few_shot.py`) | not started |
| ML→LLM→ML distillation loop | 5 | Cheap classifier replicates LLM relevance calls, LLM fallback only when uncertain | not started |
| Supervised learning | 5, 9 | Relevance classification; quality-score regression | not started |
| Vectorization/embeddings | 5, 6 | TF-IDF for distillation; sentence-transformer embeddings for FAISS | not started |
| FAISS | 6 | Semantic search, RAG retrieval backbone | not started |
| Unsupervised learning | 7 | Clustering into vibe collections | not started |
| RAG | 8 | Grounded AI summaries with cited sources | not started |
| LangChain | 8 | RAG chain orchestration | not started |
| XGBoost | 9 | Quality-score regression, tuned | not started |
| Random Forest | 9 (also 5) | Quality-score regression; distillation classifier option | not started |
| Neural networks | 9 | Small net in the quality-score bake-off | not started |
| Hyperparameter tuning | 9 | Optuna tuning XGBoost/Random Forest | not started |
| MSE / MAE | 9, 10 | Regression and forecast evaluation | not started |
| Time series | 10 | Weather-driven visit forecasting | not started |
| CrewAI multi-agent crew | 11 | Trip-planning capstone — Place Scout, Conditions Analyst, and Concierge agents collaborate, each using LangChain-wrapped tools (SQL, FAISS, weather, quality-score) | not started |
| React / TypeScript / Node.js (optional) | 14 | Next.js frontend as an alternative client for the Phase 11 FastAPI service — deployed on Vercel; bolt-on, not required for the core pipeline | not started |
