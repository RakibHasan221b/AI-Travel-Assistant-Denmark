# AI Denmark Explorer

An AI-powered place-discovery app for Copenhagen that doubles as a
deliberate showcase of the modern full-stack AI/ML toolkit — SQL,
embeddings, RAG, multi-agent orchestration, classical ML, a React/Next.js
frontend, and real deployment engineering — built, deployed, and debugged
solo, entirely on free-tier infrastructure.

**[Live app](https://ai-denmark-explorer.vercel.app/)** ·
**[Live API](https://ai-denmark-explorer-api.onrender.com/health)**

## What it does

- **Explore** — semantic ("vibe") search over 1,897 real Copenhagen places (restaurants, cafes, hotels, landmarks), each with a predicted quality score, a named vibe cluster, and an AI-grounded summary with cited sources.
- **Trip Planner** — two [CrewAI](https://www.crewai.com/) agents (Intent Analyst, Concierge) collaborate live to plan a real, honest trip recommendation, grounded in real weather and real place data — never inventing a fact it can't cite. The Intent Analyst turns a free-text request into a validated structured spec (Pydantic + `instructor`); a plain backend function then decides how each part gets searched — near/far/sequential/area — deterministically, so the LLM reasons about intent but never about spatial routing. Each recommended place carries a live `recommendation_confidence` score (XGBoost, computed fresh from real review text on every request, never a stale stored number). A live-lookup fallback (Nominatim, with Serper as a second fallback) honestly flags any place outside the curated dataset instead of pretending it has the same evidence behind it. Repeat and near-repeat requests are served from a database-backed cache instead of re-running the agents.
- **Stats Dashboard** — real aggregate SQL analytics (`GROUP BY`/`FILTER`), computed live from Postgres on every request (not pre-baked at build time), rendered as charts, plus a Model Evaluation section reporting the recommendation model's real backtested correlation against actual outcomes.

## Key results

| Model | Result |
|---|---|
| Recommendation confidence (XGBoost classifier, live per-request) | Powers the Trip Planner. Backtested against real outcomes: Pearson r = 0.713, vs r = 0.171 for the quality-score model it replaced in that path — same evaluation methodology, run on the same real places |
| Quality-score prediction (XGBoost, Optuna-tuned) | Still powers Explore's ranking. 83% of predictions within ±10 points of the true score (RMSE 8.76, R² 0.12 on 172 labeled examples — honestly reported, not inflated) |
| Weather-aware visit forecast (XGBoost, chronological split) | 97% within ±10 points (RMSE 3.39, R² 0.98) |
| RAG-summary prompt A/B test | Ran live on real places (20 GPT-4o generations, $0.027 total) with a deterministic scorer — no second LLM call needed to judge the first |

## Real engineering, not just a demo

This project was built, then actually used — and using it surfaced real bugs
that got root-caused and fixed with evidence, not guessed at:

- **Deployment**: hit and fixed 4 separate production failures getting this live on a free-tier host — a Python-version/build-sandbox mismatch, a missing transitive dependency only exposed by a narrower install, an out-of-memory kill traced to one specific import (not the framework everyone would've blamed), and a CI dependency gap. Full root-cause log: [`docs/deployment_troubleshooting.md`](docs/deployment_troubleshooting.md).
- **Data quality**: found live that a famous landmark returned zero search results (rank 701/1,896) because its only stored text was a bare Danish name — fixed with a web-enrichment pipeline, verified the fix moved it to rank 1.
- **Data quality, part 2**: every one of 1,896 places had no neighborhood assigned. Fixed via point-in-polygon matching against real Danish government district boundaries (opendata.dk + DAWA) instead of a slow, rate-limited API — 99.9% matched in 36 seconds.
- **Agent behavior**: found live that the trip-planning agent sometimes omitted weather from its answer, and padded single-place requests with irrelevant candidates — fixed both with more precise agent instructions, verified against the exact failing queries.

## Tech stack

**Data**: PostgreSQL (Neon) · pgvector · OpenStreetMap · Wikivoyage · Open-Meteo · opendata.dk / DAWA
**ML**: scikit-learn · XGBoost · Optuna · MLflow · sentence-transformers / fastembed · DistilBERT (offline sentiment scoring)
**LLM / Agents**: GPT-4o · Groq (Llama 3.3 70B) · LangChain · CrewAI · Pydantic + `instructor` (validated structured intent) · Serper (web-search fallback)
**Backend**: FastAPI (Render)
**Frontend**: React · Next.js (App Router, Server + Client Components) · TypeScript · Tailwind CSS · Recharts, deployed on Vercel
**Quality**: pytest · ruff · GitHub Actions CI

See [`docs/technique_map.md`](docs/technique_map.md) for the full technique-to-implementation mapping.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Setup | done |
| 1 | OSM place data backbone | done — 1,897 places loaded |
| 2 | ~~Seed reviews~~ | dropped, not required |
| 3 | Wikivoyage descriptions | done — 553 parsed, linked to places |
| 4 | Reddit signal pipeline (optional) | tabled — Reddit now requires Devvit developer registration + pre-approval, not just a quick form; revisit later, never blocking |
| 5 | ML→LLM→ML distillation loop | not started |
| 6 | pgvector semantic search | done — 1,897 places embedded, verified with combined category+neighborhood+semantic queries |
| 7 | Unsupervised clustering | done — 8 named clusters (silhouette-picked k), e.g. "Sushi Restaurants," "Cozy Cafes" |
| 8 | RAG-grounded summaries | done — 177 places summarized via pgvector retrieval + GPT-4o (LangChain), sources cited per summary |
| 9 | Quality-score model bake-off | done — XGBoost won (RMSE 8.76, R² 0.12, 83% within ±10pts) vs RF/Linear/NN, Optuna-tuned, MLflow-tracked, scores stored for all 1,897 places |
| 10 | Weather-aware time series | done — real Open-Meteo weather + confirmed Copenhagen event dates, chronological split (RMSE 3.39, R² 0.98, 97% within ±10pts), 5,688 place-day forecasts |
| 11 | CrewAI trip-planning crew + API | done — Intent Analyst / Concierge (Groq llama-3.3-70b), structured intent extraction with deterministic near/far/sequential/area search routing, thin FastAPI `/trip-plan`, verified live end-to-end via HTTP |
| 12 | Deployment | **live** — API on Render, frontend on Vercel. See `docs/deployment_troubleshooting.md` for the real debugging story |
| 13 | Portfolio write-up | this README, plus a CV/LinkedIn version |
| 14 | React/Next.js/TypeScript frontend | **live** — all three pages (Explore, Trip Planner, Stats Dashboard) rebuilt in Next.js (App Router, TypeScript, Tailwind, Recharts), deployed on Vercel, calling the same FastAPI backend directly from the browser |

**Bolt-on additions (not tied to the original phase numbering):** data
validation on OSM ingestion, an A/B-testing framework for RAG-summary
prompts, web-search enrichment for thin-data places, point-in-polygon
neighborhood backfill, a Trip Planner starting-location + travel-time
feature, a live-lookup fallback tool for places outside the curated dataset,
a database-backed cache that serves repeat Trip Planner requests at zero
LLM cost, a redesign from a stored quality score to a live `recommendation_confidence`
model (DistilBERT sentiment, relocated offline after a real Render memory
measurement showed the live path wouldn't fit the free tier) backtested at
r=0.713 vs the old model's r=0.171, a structured-intent architecture (Pydantic
+ `instructor`) that replaced prompt-based near/far reasoning with
deterministic backend routing, and a Model Evaluation section on the Stats
Dashboard exposing that backtest directly. 136 automated tests, GitHub
Actions CI. See [`docs/technique_map.md`](docs/technique_map.md) for full
details on each.

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
enough that combined with a long project path (e.g.
`...\deeply\nested\parent\folder\ai-denmark-explorer\.venv\...`), the
install can hit Windows' 260-character path limit. If that happens, create
the venv at a
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
`POST /trip-plan {"request": "...", "target_date": "YYYY-MM-DD", "start_location": "..."}`.

### Frontend (React/Next.js)

```bash
cd web
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_TRIP_PLANNER_API_URL, defaults to localhost:8000
npm run dev
```

Requires the API above running locally (or point `.env.local` at the live
Render URL). `npm run build` runs the same TypeScript/Next.js checks CI and
Vercel both run.

## Data sources

Free-tier, no signup friction: OpenStreetMap (bulk via Geofabrik + pyosmium
for the initial load, Nominatim for small on-demand lookups), Wikivoyage,
Open-Meteo, opendata.dk / DAWA (official Danish district boundaries). Reddit
(Phase 4) is optional and non-blocking — see `docs/architecture.md` for the
full rationale, including why the bulk and live OSM paths are split.

## Scope

Copenhagen only for now. Architecture is designed to extend to other Danish
cities later, not built to require it from day one.
