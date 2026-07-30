# AI Denmark Explorer

AI-powered place discovery, starting with a Copenhagen pilot. Combines a real,
usable product with a deliberate technique showcase (SQL, ML, RAG, agents) —
see `docs/technique_map.md` for the full mapping.

## Status

Phase 1 done — 1,468 real Copenhagen places (restaurants, cafes, hotels,
landmarks) loaded from OpenStreetMap into Postgres, with pgvector enabled
for the semantic search phase ahead.

| Phase | What | Status |
|---|---|---|
| 0 | Setup | done |
| 1 | OSM place data backbone | done — 1,468 places loaded |
| 2 | ~~Seed reviews~~ | dropped, not required |
| 3 | Wikivoyage descriptions | done — 553 parsed, 191 linked to places |
| 4 | Reddit signal pipeline (optional) | tabled — Reddit now requires Devvit developer registration + pre-approval, not just a quick form; revisit later, never blocking |
| 5 | ML→LLM→ML distillation loop | not started |
| 6 | pgvector semantic search | done — 1,468 places + 553 reviews embedded, verified with combined category+neighborhood+semantic queries |
| 7 | Unsupervised clustering | done — 18 named clusters (silhouette-picked k), e.g. "Sushi Restaurants," "Vesterbro cafes" |
| 8 | RAG-grounded summaries | done — 175 places summarized via pgvector retrieval + GPT-4o (LangChain), total cost $0.227, sources cited per summary |
| 9 | Quality-score model bake-off | done — XGBoost won (RMSE 8.76, MAE 6.76, R² 0.12, 83% of predictions within ±10pts) vs RF/Linear/NN, Optuna-tuned, MLflow-tracked, scores stored for all 1,896 places |
| 10 | Weather-aware time series | done — real Open-Meteo weather + confirmed Copenhagen event dates, chronological split (RMSE 3.39, MAE 1.99, R² 0.98, 97% within ±10pts), 5,688 place-day forecasts |
| 11 | CrewAI trip-planning crew + API | done — Place Scout / Conditions Analyst / Concierge (Groq llama-3.3-70b), thin FastAPI `/trip-plan`, verified live end-to-end via HTTP |
| 12 | Deployment | **live** — API on Render (`https://ai-denmark-explorer-api.onrender.com`), app on Streamlit Community Cloud. Three real bugs hit and fixed getting here (Python version, a missing dependency, an out-of-memory kill) — full story in `docs/deployment_troubleshooting.md` |
| 13 | Portfolio write-up | not started — natural next step now that Phase 12 is actually live |
| 14 | React/TS/Vercel frontend (optional) | not started, deprioritized |

**Bolt-on additions (not tied to the original phase numbering):** data validation on OSM ingestion, an A/B-testing framework for RAG-summary prompts (run live: a=3, b=2, tie=5), and web-search enrichment for places with thin/no data (fixed a real bug live — the Little Mermaid statue went from search rank 701/1,896 to rank 1). See `docs/technique_map.md` for full details on each.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
cp .env.example .env     # fill in DATABASE_URL, GROQ_API_KEY
psql $DATABASE_URL -f db/schema.sql
pytest
```

### Phase 11's `agent` extra: use a short-path venv on Windows

`pip install -e ".[dev,embeddings,rag,agent]"` pulls in `crewai`, which
depends on `torch` — and torch's own bundled license files are nested deep
enough that combined with a long project path (e.g. this repo's, under
`...\CLAUDE PROJECTS\ai-denmark-explorer\.venv\...`), the install can hit
Windows' 260-character path limit. If that happens, create the venv at a
short path outside the project instead, e.g.:

```bash
python -m venv C:\Users\<you>\.venvs\ade
C:\Users\<you>\.venvs\ade\Scripts\python.exe -m pip install -e ".[dev,embeddings,rag,agent]"
```

Also install this extra in its own venv, not one shared with other
projects — `crewai` requires `langchain-core>=1.0`, which conflicts with the
`langchain 0.3.x` pin used for Phase 8's `rag` extra elsewhere on the same
machine.

Run the API: `uvicorn api.main:app --port 8000`, then
`POST /trip-plan {"request": "...", "target_date": "YYYY-MM-DD"}`.

## Data sources

Free-tier, no signup friction: OpenStreetMap (bulk via Geofabrik + pyosmium
for the initial load, Nominatim for small on-demand lookups), Wikivoyage,
Open-Meteo, opendata.dk. Reddit (Phase 4) is optional and non-blocking —
see `docs/architecture.md` for the full rationale, including why the bulk
and live OSM paths are split.

## Scope

Copenhagen only for now. Architecture is designed to extend to other Danish
cities later, not built to require it from day one.
