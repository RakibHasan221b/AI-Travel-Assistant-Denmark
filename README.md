# AI Denmark Explorer

An AI-powered place-discovery app for Copenhagen, built, deployed, and
debugged solo, entirely on free-tier infrastructure.

It's also a full-stack AI/ML case study: SQL, embeddings, RAG, multi-agent
orchestration, classical ML, and a React/Next.js frontend, all shipped
end to end, not just prototyped.

**[Live app](https://ai-denmark-explorer.vercel.app/)** ·
**[Live API](https://ai-denmark-explorer-api.onrender.com/health)**

## What it does

- **Explore**: semantic ("vibe") search over 1,897 real Copenhagen places (restaurants, cafes, hotels, landmarks), each with a predicted quality score, a named vibe cluster, and an AI-grounded summary with cited sources.
- **Trip Planner**: two [CrewAI](https://www.crewai.com/) agents (Intent Analyst, Concierge) plan a real, honest trip recommendation, grounded in real weather and real place data, never inventing a fact it can't cite.
  - The Intent Analyst turns free text into a validated structured spec (Pydantic + `instructor`); a plain backend function decides how each part gets searched (near/far/sequential/area), so the LLM reasons about intent but never about spatial routing.
  - Every recommended place carries a live `recommendation_confidence` score (XGBoost), computed fresh from real review text on every request, never a stale stored number.
  - A live-lookup fallback (Nominatim, then Serper) honestly flags any place outside the curated dataset instead of pretending it has the same evidence behind it.
  - Repeat and near-repeat requests are served from a database-backed cache instead of re-running the agents.
- **Stats Dashboard**: real aggregate SQL analytics (`GROUP BY`/`FILTER`), computed live from Postgres on every request (not pre-baked at build time), rendered as charts, plus a Model Evaluation section reporting the recommendation model's real backtested correlation against actual outcomes.

## Architecture

![AI Denmark Explorer architecture diagram](docs/images/architecture.png)

One request pipeline, six steps: a natural-language request becomes a
validated structured intent, gets routed and searched deterministically,
ranked by a real ML model, and answered with a grounded response. The LLM
reasons about what the traveler means, never about where to search or
what's true.

**Request flow**: Natural Language Request → Structured Intent → Validated
Schema → Deterministic Routing → Search + ML Ranking → Grounded Response.

**Data.** Real Copenhagen places (1,897 and counting) come from
OpenStreetMap, bulk-loaded via `osmium` and enriched on-demand through
Nominatim. Wikipedia/Wikivoyage supply place descriptions; Serper (web
search) fills in the rest for places with thin data, with every candidate
independently re-verified before it's trusted. Every incoming record is
validated with Pydantic and neighborhood-matched via real point-in-polygon
geometry (Shapely) against official Danish district boundaries, not a
slow, rate-limited geocoding API.

**Search & database.** PostgreSQL (Neon, scale-to-zero) with pgvector
stores every place and its embedding (`all-MiniLM-L6-v2`). Semantic search
retrieves by meaning, then a reranking layer combines that similarity with
real lexical name-matching so an exact-name query doesn't get buried under
topically-similar decoys. Spatial relationships (near an anchor, far from
it, a sequence, a neighborhood constraint) are resolved by explicit
backend routing, never inferred by the LLM.

**Machine learning.** A `recommendation_confidence` score is computed live
for every recommended place: DistilBERT's offline-precomputed sentiment
reading and a freshly-computed MiniLM semantic-nuance signal both feed a
trained XGBoost classifier. Backtested against real outcomes at Pearson
r = 0.713, more than 4x the correlation of the quality-score model it
replaced in this path (r = 0.171). Optuna tunes hyperparameters; a
separate, Explore-only XGBoost model still handles the original
quality-score ranking task.

**Generative AI.** Two CrewAI agents. The Intent Analyst (OpenAI
`gpt-4o-mini`) turns free text into a validated structured spec; it has
no database tools at all. The Concierge (OpenAI `gpt-4o`) narrates the
final answer from real retrieved data, with an explicit guard against
claiming proximity or facts the pipeline didn't actually establish.
LangChain runs the separate RAG-summary pipeline that produces Explore's
cited place summaries.

**Live services.** Open-Meteo (real weather for the actual requested date,
not a fixed lookup table), Serper (evidence search), and Nominatim (live
place lookup) run alongside the main search path, each with its own
honest fallback if it comes back empty.

**Application layer.** FastAPI backend on Render, Next.js/React/TypeScript
frontend on Vercel, calling the API directly from the browser.

**Quality.** 144 automated tests, GitHub Actions CI, `ruff` linting.

## Key results

| Model | Result |
|---|---|
| Recommendation confidence (XGBoost classifier, live per-request) | Powers the Trip Planner. Backtested against real outcomes: Pearson r = 0.713, vs r = 0.171 for the quality-score model it replaced in that path (same evaluation methodology, run on the same real places) |
| Quality-score prediction (XGBoost, Optuna-tuned) | Still powers Explore's ranking. 83% of predictions within ±10 points of the true score (RMSE 8.76, R² 0.12 on 172 labeled examples, honestly reported, not inflated) |
| Weather-aware visit forecast (XGBoost, chronological split) | 97% within ±10 points (RMSE 3.39, R² 0.98) |
| RAG-summary prompt A/B test | Ran live on real places (20 GPT-4o generations, $0.027 total) with a deterministic scorer; no second LLM call needed to judge the first |

## Hardened by real use

This wasn't verified once and shipped; it was built, then actually used,
and using it surfaced real problems that got root-caused with evidence and
fixed at the architecture level, not patched over:

- **A real LLM reliability bug became a deterministic-routing redesign.** Live testing found that a bare sequence word like "then" (as in "...and then grab coffee") could get misread as a spatial-relationship marker, sending the agent searching "near" a place that was never meant as an anchor. Rather than patching the prompt again, spatial routing was moved out of the LLM entirely: a validated backend function now decides near/far/sequential/area deterministically, so the model reasons about intent and never about where to search.
- **Even that deterministic "near" search had a real relevance bug.** Asking for "a sushi place near the Little Mermaid" silently returned the closest restaurant of any kind, Italian included, because `near` ranked candidates by distance alone and nothing ever read what was actually being asked for. Rather than hardcoding a cuisine list, which can never cover what a user might type next, the fix reused the same semantic-relevance ranking Explore already had: the traveler's own wording is now scored against every nearby candidate, and if genuinely nothing matches, the app says so honestly and falls back to a live search instead of quietly substituting something else. Verified with 8 new regression tests and real before/after checks against the live database.
- **Four separate production failures, each root-caused to a specific line, not blamed on the framework.** Getting this live on a free-tier host surfaced a Python-version/build-sandbox mismatch, a missing transitive dependency only exposed by a narrower install, an out-of-memory kill traced to one specific import, and a CI dependency gap.
- **A famous landmark returned zero search results, in production.** The Little Mermaid statue's only stored text was a bare Danish OSM tag, traced to rank 701 out of 1,896 in semantic search. Fixed with a web-enrichment pipeline; verified the fix moved it to rank 1.
- **Every place in the database was missing its neighborhood.** `addr:suburb` is rarely set on individual OSM points. Fixed with real point-in-polygon matching against official Danish district boundaries (opendata.dk + DAWA) instead of a slow, rate-limited API; 99.9% matched in 36 seconds.

## Tech stack

| Tool | Category | What it does here |
|---|---|---|
| PostgreSQL (Neon) | Data | Primary database, scale-to-zero free tier |
| pgvector | Data | Vector similarity search, in the same SQL query as structured filters |
| OpenStreetMap | Data | Source of every real place record (1,897 and counting) |
| osmium | Data | Bulk OSM extraction for the initial load |
| Nominatim | Data | Live geocoding and single-place lookup |
| Wikivoyage / Wikipedia | Data | Place descriptions |
| opendata.dk / DAWA | Data | Official Danish district boundaries for neighborhood matching |
| Shapely | Data | Point-in-polygon neighborhood matching against those boundaries |
| Serper | Data / Agents | Web-search fallback for thin-data places, independently re-verified before trusting |
| scikit-learn | ML | Unsupervised clustering, cross-validation |
| XGBoost | ML | Quality-score and recommendation-confidence classifiers |
| Optuna | ML | Hyperparameter tuning |
| MLflow | ML | Experiment tracking |
| sentence-transformers / fastembed | ML | `all-MiniLM-L6-v2` embeddings, semantic search + recommendation signal |
| DistilBERT | ML | Offline sentiment scoring (`transformers`), feeds the recommendation classifier |
| GPT-4o | Agents | Concierge narration and RAG-summary generation |
| GPT-4o-mini | Agents | Intent Analyst's structured-intent extraction |
| CrewAI | Agents | Multi-agent orchestration (Intent Analyst, Concierge) |
| Pydantic + `instructor` | Agents | Validated structured-intent extraction from free text |
| LangChain | Agents | RAG retrieval chain behind Explore's cited place summaries |
| Open-Meteo | Live services | Real weather for the actual requested date, archive + forecast |
| FastAPI | Backend | API framework, deployed on Render |
| React / Next.js | Frontend | App Router, Server + Client Components, deployed on Vercel |
| TypeScript | Frontend | Type safety across the frontend |
| Tailwind CSS | Frontend | Styling |
| Recharts | Frontend | Stats Dashboard charts |
| pytest | Quality | 144 automated tests |
| ruff | Quality | Linting |
| GitHub Actions | Quality | CI on every push |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
cp .env.example .env     # fill in DATABASE_URL, OPENAI_API_KEY
psql $DATABASE_URL -f db/schema.sql
pytest
```

### The `agent` extra: use a short-path venv on Windows

`pip install -e ".[dev,embeddings,rag,agent]"` pulls in `crewai`, which
depends on `torch`, and torch's own bundled license files are nested deep
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
projects; `crewai` requires `langchain-core>=1.0`, which conflicts with the
`langchain 0.3.x` pin used elsewhere on the same machine.

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

## Scope

Copenhagen only for now. Architecture is designed to extend to other Danish
cities later, not built to require it from day one.
