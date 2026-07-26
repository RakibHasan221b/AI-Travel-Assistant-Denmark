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
| 8 | RAG-grounded summaries | not started |
| 9 | Quality-score model bake-off | done — XGBoost won (MSE 100.9) vs RF/Linear/NN, Optuna-tuned, MLflow-tracked, scores stored for all 1,468 places |
| 10 | Weather-aware time series | done — real Open-Meteo weather + confirmed Copenhagen event dates, chronological split (MSE 17.06, MAE 2.35), 10,276 place-day forecasts |
| 11 | CrewAI trip-planning crew + API | not started |
| 12 | Deployment | not started |
| 13 | Portfolio write-up | not started |
| 14 | React/TS/Vercel frontend (optional) | not started, deprioritized |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
cp .env.example .env     # fill in DATABASE_URL, GROQ_API_KEY
psql $DATABASE_URL -f db/schema.sql
pytest
```

## Data sources

Free-tier, no signup friction: OpenStreetMap (bulk via Geofabrik + pyosmium
for the initial load, Nominatim for small on-demand lookups), Wikivoyage,
Open-Meteo, opendata.dk. Reddit (Phase 4) is optional and non-blocking —
see `docs/architecture.md` for the full rationale, including why the bulk
and live OSM paths are split.

## Scope

Copenhagen only for now. Architecture is designed to extend to other Danish
cities later, not built to require it from day one.
