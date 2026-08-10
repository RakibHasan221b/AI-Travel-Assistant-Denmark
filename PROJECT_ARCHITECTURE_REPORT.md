# AI Denmark Explorer — Trip Planner: Current Architecture Report

**Status: investigation only. No files were edited, no commits or pushes were made while producing this report.**
Every claim below is traced to a specific file/function/line from the repository as it exists on branch `main` (commit `07d2259`) with the currently-uncommitted working-tree changes to `agent/crew.py`, `agent/tools.py`, and `tests/test_crew.py` (see §14). Where something could not be verified by reading the code, it is explicitly marked "not verified from the current code."

---

## 1. Overall architecture

```
┌─────────────┐      HTTPS       ┌──────────────────┐
│  Next.js     │ ───────────────► │  FastAPI (api/    │
│  frontend    │ ◄─────────────── │  main.py)          │
│  (Vercel)    │      JSON        │  Render, 1 process │
└─────────────┘                  └─────────┬──────────┘
                                            │
                         ┌──────────────────┼───────────────────────┐
                         ▼                  ▼                       ▼
                 ┌───────────────┐  ┌───────────────┐      ┌────────────────┐
                 │ agent/crew.py  │  │ agent/tools.py │      │ api/ranking.py  │
                 │ CrewAI: Place  │  │ pgvector search,│      │ deterministic   │
                 │ Scout +        │◄─┤ weather, live   │      │ lexical rerank  │
                 │ Concierge      │  │ discovery, DB   │      │ (no DB/network) │
                 │ (OpenAI        │  │ helpers         │      └────────────────┘
                 │ gpt-4o-mini)   │  └───────┬─────────┘
                 └───────┬────────┘          │
                         │                   ▼
                         │           ┌──────────────────┐
                         │           │ PostgreSQL +      │
                         │           │ pgvector           │
                         │           │ (places, reviews,  │
                         │           │  ml_predictions,    │
                         │           │  weather_daily,      │
                         │           │  trip_plan_cache)     │
                         │           └──────────┬───────────┘
                         │                      │
                         ▼                      ▼
               ┌──────────────────┐   ┌────────────────────┐
               │ agent/recommend-  │   │ Serper (via         │
               │ ation_service.py  │   │ ingestion/web_       │
               │ XGBoost Booster + │   │ enrichment.py) +      │
               │ numpy MiniLM      │   │ Wikipedia REST API     │
               │ sigmoid           │   │ — evidence enrichment  │
               └──────────────────┘   │ for ONE named place,   │
                                       │ not candidate discovery │
                                       └────────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │ Open-Meteo         │
               │ (weather_daily      │
               │  cache + live call)  │
               └──────────────────┘
```

Services and where each lives in the repo:
- **Frontend**: `web/` — Next.js App Router, deployed on Vercel. Trip Planner, Explore, Stats Dashboard, a shared `NavBar`.
- **Backend API**: `api/main.py` — FastAPI, one process on Render (`uvicorn api.main:app`, no `--workers`, confirmed by the memory-sharing rationale documented in `agent/tools.py:31-36`).
- **Database**: PostgreSQL with the `pgvector` and `pgcrypto` extensions (`db/schema.sql:1-5`).
- **CrewAI**: `agent/crew.py` — two `Agent`s (Place Scout, Concierge), OpenAI `gpt-4o-mini`.
- **OpenAI**: accessed through `crewai.llms.providers.openai.completion.OpenAICompletion` (native provider, not litellm — `agent/crew.py:18-20`), reading `OPENAI_API_KEY`.
- **Pydantic**: `agent/crew.py`'s `PlaceRecommendation`/`TripPlanOutput` (internal schema, used by CrewAI's `output_pydantic` + `instructor`), and `api/main.py`'s independently-declared mirror classes (`TripPlanRequest`, `PlaceRecommendation`, `TripPlanResponse`, plus `Explore*`/`Stats*` models).
- **XGBoost**: `agent/recommendation_service.py` — native `xgboost.Booster` (not the sklearn wrapper).
- **Serper**: `ingestion/web_enrichment.py` (`search_web`/`filter_results`), called from `agent/tools.py`'s `_search_place_evidence`.
- **Weather**: Open-Meteo, called from `agent/tools.py`'s `_fetch_and_cache_live_weather`/`_weather_structured`, cached in `weather_daily`.
- **Other**: Nominatim (free geocoding, `agent/crew.py:geocode` and `agent/tools.py:search_place_live`), Wikipedia REST API (`agent/tools.py:_wikipedia_summary`).

Two docs already exist in the repo — `docs/architecture.md` and `docs/PROJECT_STATUS.md` — but this report is built from reading the code directly, not from those docs, per the instruction to avoid guessing.

---

## 2. Complete Trip Planner journey

Trace for: *"I want to see the Little Mermaid and have coffee nearby afterwards."*

1. **Frontend sends**: `web/components/trip-planner/TripPlannerClient.tsx:21-49` calls `planTrip()` (`web/lib/api.ts:27-39`) — a plain `fetch(POST /trip-plan)`, body `{request, target_date, start_location}`. Deliberately not proxied through a Vercel API route because the request can take 1-2 minutes (three sequential LLM calls), which would exceed Vercel's serverless timeout (`web/lib/api.ts:23-26`).
2. **Endpoint**: `POST /trip-plan` in `api/main.py:91`.
3. **Handler function**: `trip_plan()` (`api/main.py:91-120`), which calls `agent.crew.plan_trip()` (`api/main.py:96`).
4. **plan_trip()** (`agent/crew.py:1124-1187`):
   - Checks `trip_plan_cache` for an exact match (`_get_exact_cache`, `agent/crew.py:569-587`) — zero LLM cost on a hit.
   - Geocodes `start_location` via Nominatim (`geocode()`, `agent/crew.py:151-166`) and stores it in a module-level global via `set_trip_start()` (`agent/tools.py:65-69`).
   - If the same request+date exists in cache under a different start location, reuses it and only recomputes travel time (`_recompute_travel`, `agent/crew.py:621-645`) — no LLM call.
   - Otherwise resets `agent/tools.py`'s per-request tool-call cache (`reset_tool_call_cache()`) and the LLM-call counter, then calls `build_crew().kickoff(inputs=...)`.
5. **CrewAI starts**: `build_crew()` (`agent/crew.py:352-536`) constructs a `Crew` with `Process.sequential` and two tasks.
6. **Agents**:
   - **Place Scout** (`agent/crew.py:357-368`) — tools: `search_places`, `search_places_near`, `top_quality_places`, `search_place_live`.
   - **Concierge** (`agent/crew.py:370-384`) — tools: `place_details`, `travel_time_estimate`, `weather_conditions`.
7. **Tools available**: all six are `agent/tools.py` functions decorated `@tool` + `@_cache_tool_calls` (per-request memoization, `agent/tools.py:105-120`).
8. **How the request is interpreted**: entirely by the Scout's LLM reasoning against `scout_task`'s natural-language `description` (`agent/crew.py:386-449`) — there is no structured intermediate representation. The Scout decides, per part of the sentence, which tool to call and with what arguments purely from following that prose. For this example: it should call `search_places` (or similar) for "Little Mermaid" as a named place, then `search_places_near(anchor_place="Den lille Havfrue", category="cafe")` for "coffee nearby afterwards" because the prompt tells it explicit proximity wording (`nearby`) beats the sequence word (`afterwards`).
9. **Places searched — pgvector**: `search_places()` embeds the query with `get_embed_model()` (fastembed/ONNX MiniLM) and runs `ORDER BY embedding <=> qvec` (`agent/tools.py:255-296`), then reranks with `api/ranking.py`'s deterministic lexical layer.
10. **XGBoost involved**: not in the Scout step. It runs later, inside `place_details()` → `_lookup_place_structured()` → `predict_recommendation()` (`agent/tools.py:907-952` → `agent/recommendation_service.py:176-190`).
11. **Distances calculated**: `haversine_km()` (`agent/tools.py:171-177`), pure math, never LLM-computed.
12. **Near/far relationship determined**: at scout time, purely by the LLM following `scout_task`'s prose rules (§3, §9 below cover the exact current rules and their weakness). After the crew finishes, `agent/crew.py`'s `_reconcile_near_relationships()` (line 678) independently re-derives the real anchor and real distance from the Scout's *actual* `search_places_near` tool-call arguments (via `get_cached_tool_calls`, not by trusting the Concierge's prose), and overwrites `near_place`/`near_distance_km` — this is the deterministic backend correction layer already in place.
13. **Weather obtained**: the Concierge calls `weather_conditions(target_date)` once (`agent/crew.py:466-468`), which calls `_weather_structured()` (`agent/tools.py:1183-1246`) — checks `weather_daily` cache first, falls back to a live Open-Meteo call.
14. **Pydantic validates the result**: the Concierge's task has `output_pydantic=TripPlanOutput` (`agent/crew.py:528`) — CrewAI uses the `instructor` library to constrain/validate the LLM's structured JSON answer against `TripPlanOutput`/`PlaceRecommendation` (`agent/crew.py:180-289`), including several `field_validator`s that repair known LLM output quirks (null category, null sources, placeholder text like "unknown" leaking into fields).
15. **JSON reaching the frontend**: `plan_trip()` returns `TripPlanOutput.model_dump()` after running it through `_ensure_start_distance(_reconcile_near_relationships(result))` (`agent/crew.py:1184`) and saving it to `trip_plan_cache`. `api/main.py`'s `trip_plan()` wraps that dict into its own `TripPlanResponse` Pydantic model (`api/main.py:75-88, 120`) and FastAPI serializes it to JSON.
16. **Frontend renders**: `TripPlannerClient` stores the response in state and renders `TripPlanResults` (`web/components/trip-planner/TripPlanResults.tsx`), which maps `result.places` to `PlaceCard` (`web/components/trip-planner/PlaceCard.tsx`). `PlaceCard.buildTravelLine()` (lines 19-34) already prioritizes `near_place`/`near_distance_km` over `distance_km` — this file requires **no changes** for the near/far problem; the bug has always been a backend data-correctness problem, not a display problem.

---

## 3. Natural language understanding

There is **no structured intent representation** anywhere in the current system. The entire chain from raw user sentence to tool calls is: user text → inserted verbatim into `scout_task.description`'s `{request}` placeholder (`agent/crew.py:388`) → the Scout LLM reads a large block of natural-language *rules* (not a schema) and decides, in one reasoning pass, which of `search_places` / `search_places_near` / `top_quality_places` / `search_place_live` to call and with what string arguments. The current rules (all currently live in `agent/crew.py:398-433`, restructured most recently to fix the "far away" regression) encode an explicit priority order:

1. Explicit distance/far wording ("far away", "far from", "elsewhere", ...) → `search_places`, never `search_places_near`, even if a sequence word or named place also appears.
2. Explicit proximity wording ("near", "nearby", "within X km", "walking distance", ...) naming another place anywhere in the request → `search_places_near(anchor_place=..., max_km=...)`.
3. Neither — including a part introduced *only* by a bare sequence word ("then", "afterwards") — → treated as no reference point, `search_places` only.

Conceptual test against the required phrases (★ = deterministically enforced once the tool is chosen; the tool *choice* itself is always LLM judgment, never enforced):

| Phrase | Current expected tool choice | Deterministic backend behavior once chosen |
|---|---|---|
| "coffee nearby" | `search_places_near` (rule 2) | ★ real haversine distance from resolved anchor |
| "coffee near the Little Mermaid" | `search_places_near(anchor_place="Little Mermaid")` | ★ same |
| "coffee within 1 km of the Little Mermaid" | `search_places_near(..., max_km=1)` | ★ `_coerce_max_km` clamps/validates (`agent/tools.py:212-222`), `_places_near` enforces the radius server-side (`agent/tools.py:332`) |
| "coffee far away from the Little Mermaid" | `search_places` (rule 1) | ★ no anchor is ever computed; place gets `near_place=None` |
| "coffee afterwards" (bare) | `search_places` (rule 3) | — |
| "coffee afterwards nearby" | `search_places_near` (rule 2, "nearby" wins) | ★ |
| "coffee before visiting the museum" | Not covered by any explicit rule — "before" is a sequence word like "afterwards"/"then" but is not named anywhere in the current prompt text. Behavior is undefined/untested. | — |
| "something around Nørrebro" | Not modeled at all — "around Nørrebro" is a **neighborhood constraint**, not a "near a named place" relationship. The Scout has no tool parameter for this; `search_places` doesn't accept free-text location scoping beyond its own `neighborhood=` exact-match parameter, which the Scout would have to guess maps to "Nørrebro" from the sentence. | — |
| "a restaurant near me" | Ambiguous — "me" is the traveler's own `start_location`, not another named place. `search_places_near` requires a resolvable place name (`_resolve_place`); "me" is not one. Current prompt does not address this case at all. | — |

**What's LLM vs. deterministic today**: tool *selection* (which relationship applies) and tool *argument extraction* (which place is the anchor, what `max_km` to pass) are 100% LLM judgment, done in a single pass, governed only by prose instructions. Once a tool is actually called, everything downstream is deterministic: `_places_near`'s radius filter, `_reconcile_near_relationships`'s re-derivation of the real anchor/distance from the *actual* recorded tool-call arguments (not the LLM's transcription of them), and `_ensure_start_distance`'s backfill of the primary place's own start distance.

**Where this is fragile, confirmed by real testing this session** (see §12, §13): a real end-to-end call for *"I want to see the Little Mermaid and then have coffee."* (bare "then", no proximity or distance wording at all) still resulted in the Scout calling `search_places_near(anchor_place="the Little Mermaid", category="cafe", ...)` — directly contradicting rule 3 above, which explicitly uses this exact sentence as its own non-triggering worked example. This is not a wording-ambiguity bug in the prompt (the text was reread and is unambiguous); it is evidence that a single LLM reasoning pass over prose rules does not reliably enforce a decision boundary, no matter how explicitly the boundary is worded. This is the concrete, reproduced evidence behind the user's decision to require a structural (non-prompt) fix.

---

## 4. Pydantic models

| File | Class | Purpose | Key fields | Created | Validated | Consumed |
|---|---|---|---|---|---|---|
| `agent/crew.py:180` | `PlaceRecommendation` | CrewAI's internal per-place schema | `name, category, neighborhood, recommendation_confidence, recommendation_label, vibe_cluster, summary, sources, distance_km, walk_minutes, bike_minutes, travel_note, near_place, near_distance_km, why_recommended` | Instantiated by `instructor` from the Concierge's LLM output, or manually in `deterministic_trip_plan`/`_compound_deterministic_places`/`_trip_plan_from_cached_results` | 3 `field_validator`s repair null category/neighborhood, null sources, and placeholder text ("unknown" etc.) leaking into fields (lines 226-271) | `TripPlanOutput.places` |
| `agent/crew.py:274` | `TripPlanOutput` | Whole crew output | `places: list[PlaceRecommendation], weather_summary, overall_note` | `output_pydantic=TripPlanOutput` on `concierge_task` (line 528); `instructor` enforces this against the raw LLM completion | Pydantic's own validation | `plan_trip()` returns `.model_dump()` of this |
| `api/main.py:50` | `TripPlanRequest` | Inbound `/trip-plan` body | `request, target_date, start_location` | FastAPI request parsing | Pydantic | `trip_plan()` handler |
| `api/main.py:56` | `PlaceRecommendation` (separate class, same name) | Outbound per-place shape | Same field set as the crew's version, declared independently on purpose ("this module shouldn't need to know about crew.py's internal schema class" — line 76-78 comment) | `TripPlanResponse(**result)` (line 120) | Pydantic | Frontend JSON |
| `api/main.py:75` | `TripPlanResponse` | Outbound `/trip-plan` body | `places, weather_summary, overall_note` | line 120 | Pydantic | Frontend |
| `api/main.py:123,135` | `ExplorePlaceResult`, `ExploreResponse` | `/explore` response | `name, category, similarity, quality_score, vibe_cluster, summary, sources, ...` | `explore()` handler | Pydantic | Explore page |
| `api/main.py:233-273` | `NeighborhoodQuality`, `CategoryQuality`, `VibeClusterSize`, `RatedAspect`, `ForecastPoint`, `StatsResponse` | `/stats` response | aggregate rows | `stats()` handler | Pydantic | Stats Dashboard page |

**Example JSON — current request model** (`POST /trip-plan`):
```json
{ "request": "I want to see the Little Mermaid and have coffee nearby afterwards.",
  "target_date": "2026-09-01", "start_location": "Vanløse" }
```

**Example JSON — current response model** (one place, abbreviated):
```json
{
  "places": [
    { "name": "Terminalen kaffebar - Seaside Toldboden", "category": "cafe",
      "neighborhood": "Indre By", "recommendation_confidence": 78, "recommendation_label": "recommended",
      "distance_km": null, "near_place": "Den lille Havfrue", "near_distance_km": 0.32,
      "why_recommended": "A quick walk from the Little Mermaid, ..." }
  ],
  "weather_summary": "2026-09-01: high 19°C / low 12°C, 0.0mm precipitation, 14km/h wind.",
  "overall_note": "..." }
```

**No model in the codebase today represents "user intent" as a structured object** — there is no `TripSpecification`, `ItineraryRequest`, or similar. This is the concrete gap the user's proposed architecture (§16) targets.

---

## 5. PostgreSQL and pgvector

Full schema at `db/schema.sql`. Relevant tables for the Trip Planner:

- **`places`** (`db/schema.sql:7-31`): `place_id, osm_id, name, category, subcategory, lat, lon, address, neighborhood, opening_hours, osm_tags jsonb, price_level, embedding vector(384), data_status ('curated'|'live_discovered'), source_url`. HNSW index on `embedding` (`vector_cosine_ops`), plus btree on `category`/`neighborhood` and GIN on `osm_tags`.
- **`reviews_raw`**: raw review/description text, `source_type` CHECK-constrained to `seed|wikivoyage|wikipedia|reddit_post|reddit_comment|web_search`, own `embedding vector(384)` for RAG retrieval.
- **`place_mentions`**: links `reviews_raw` rows to `places`.
- **`aggregated_sentiment`**: per-place, per-aspect rollup (food/service/ambiance/value/location/overall), shown in `place_details`'s "Rated aspects" line.
- **`ai_summaries`**: RAG-grounded summary text + `sources jsonb`.
- **`ml_predictions`**: generic `target`/`predicted_value` table — holds **both** the old `quality_score` target (used by `top_quality_places`) and the `distilbert_sentiment` target (used as an XGBoost feature). These are two different, independently-computed numbers stored in the same table under different `target` strings.
- **`weather_daily`**: Open-Meteo cache, `date` primary key.
- **`visit_time_forecast`**: Outdoor Interest Index.
- **`trip_plan_cache`**: request/date/start_location → cached full `TripPlanOutput` JSON, 24h TTL enforced at query time (`agent/crew.py:576-586`).

**Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim. Batch ingestion (offline, not in the live request path) uses real `sentence-transformers`; the live API path uses `fastembed`'s ONNX export of the same weights (`get_embed_model()`, `agent/tools.py:183-187`) — verified cosine similarity 1.0000 between the two per the module's own docstring (line 25), adopted specifically to avoid a `torch` import in the live process (a real, previously-confirmed Render OOM cause).

**Semantic search**: `ORDER BY embedding <=> %(qvec)s` (cosine distance operator from pgvector), e.g. `agent/tools.py:288`, `api/main.py:193`.

**Category filtering**: exact-match `WHERE category = %(category)s` (`agent/tools.py:283`, `_places_near`'s `category` param at `agent/tools.py:338-340`).

**Neighborhood filtering**: exact match in `_search_places_rows` (`agent/tools.py:285-287`); `ILIKE '%...%'` (substring) in `/explore` (`api/main.py:190-192`) — these two are inconsistent with each other (exact vs. substring), not verified whether intentional.

**Candidate retrieval**: `search_places`/`_search_places_rows` pulls `DEFAULT_POOL_SIZE=40` (`api/ranking.py:67`) candidates by raw vector distance, then `rank_explore_candidates()` reranks with a combined `similarity + 0.45*name_match_score + category_intent_score` and filters by `RELEVANCE_FLOOR=0.40`/`RELEVANCE_GAP=0.30` (`api/ranking.py:58-59`) before capping to the caller's `limit` — this can legitimately return **fewer** results than requested, or zero, by design (not a bug).

**"Near X" retrieval**: `_places_near()` (`agent/tools.py:315-348`) is a *separate* code path — it does not use pgvector or embeddings at all. It pulls every place matching an optional `category` filter, computes real `haversine_km` distance to the anchor's coordinates in Python, sorts, and filters to `<= radius` (capped at `MAX_NEARBY_KM=2.0`). This is real geographic proximity, not semantic similarity.

**How a structured itinerary spec could translate to DB searches (not yet implemented, described for §16)**: a `primary` field with a named place would resolve via `_resolve_place`/`_search_places_rows` exactly as today; a `secondary` with `relation=near, anchor=X` would call `_places_near` directly with the anchor's already-resolved coordinates (skipping the LLM-mediated `search_places_near` tool call entirely for the *decision* of whether to search near — the LLM would only have supplied the anchor name and relation type, not chosen the tool); `relation=far` would skip `_places_near` outright and go straight to `search_places`/`_search_places_rows`; `neighborhood` constraints would map onto `_search_places_rows`'s existing `neighborhood` parameter, which is unused by the Scout today.

---

## 6. XGBoost

This is the live recommendation scoring system (`agent/recommendation_service.py`), which is **separate from and more recent than** the older `quality_score` still stored in `ml_predictions`.

1. **Model used**: `pipeline/modeling/recommendation_classifier.json` — a trained XGBoost model with a `binary:logistic` objective, loaded via the **native** `xgboost.Booster` API (`agent/recommendation_service.py:74-79`), specifically *not* `XGBClassifier` because that sklearn-compatible wrapper hard-requires `scikit-learn` to be importable (confirmed in the module's own docstring, lines 27-39) — importing `sklearn` alone measured ~94MB RSS, which combined with FastEmbed + a built CrewAI crew pushed the live process over Render's 512MB free-tier limit in a real, previously-confirmed OOM incident.
2. **Where loaded**: `_get_xgb_model()` (`agent/recommendation_service.py:74-79`) — a lazy, module-level singleton, loaded once on first use, kept in memory for the process lifetime (same pattern as `get_embed_model()` in `agent/tools.py`).
3. **Features used** (`extract_features()`, `agent/recommendation_service.py:116-173`), per `recommendation_feature_schema.json`'s `feature_order`:
   - `distilbert_sentiment` — **precomputed offline** (`pipeline/modeling/precompute_distilbert_sentiment.py`), read from `ml_predictions` (`target='distilbert_sentiment'`) by `agent/tools.py:852,929`, passed in as `place["distilbert_sentiment_score"]`. Never computed live (DistilBERT alone measured ~804MB RSS live — too expensive).
   - `minilm_good_probability` — computed live, `sigmoid(embedding @ coef_.T + intercept_)` using the shared `fastembed` MiniLM embedding and exported `LogisticRegression` weights (`.npz`, no live sklearn import — `_minilm_good_probability`, lines 101-113). Falls back to `0.5` (neutral) when there is no review text at all.
   - Structural one-hot/boolean features: `has_subcategory, has_opening_hours, has_phone, has_website, has_price_level, price_level, review_count`, plus one-hot `cat_<category>` columns.
   - **Deliberately excludes** any stored rating — using the training label as a feature would leak the answer (comment, line 126-127).
4. **Prediction**: `predict_recommendation()` (`agent/recommendation_service.py:176-190`) builds the feature vector, calls `model.predict(xgb.DMatrix(vector))[0]` — because the model was trained with `binary:logistic`, this already returns the sigmoid-transformed probability of the positive class (verified numerically, diff = 0.0, per the docstring at line 33-36), equivalent to what `predict_proba` would give from the same file.
5. **What the score means**: `recommendation_probability` (0-1) → `recommendation_confidence` = `round(probability*100)` **only if** the place has real review text (`has_review_text` signal) — otherwise both `recommendation_confidence` and `recommendation_label` are explicitly `None`, never a guessed/default value (`agent/tools.py:935-940`). `label` is `"recommended"` if `probability >= 0.5` else `"not recommended"`. This is explicitly documented (`agent/crew.py:186-198`) as "how likely is this place to be a good recommendation," **not** an objective quality rating — this is a real, deliberate naming/framing decision made after evaluation showed the old `quality_score` (structured metadata only) correlated weakly with outcomes (r=0.171) versus this model's r=0.713.
6. **How it reaches the frontend**: `predict_recommendation()` → `_lookup_place_structured()` builds `recommendation_confidence`/`recommendation_label`/`confidence_unavailable_reason` (`agent/tools.py:960-973`) → either formatted into `place_details()`'s text output (LLM path, `agent/tools.py:1015-1019`) or returned as a dict directly (deterministic path) → `agent/crew.py`'s `PlaceRecommendation.recommendation_confidence/recommendation_label` → `TripPlanOutput` → `plan_trip()`'s dict → `api/main.py`'s mirrored `PlaceRecommendation` → JSON → `web/lib/types.ts`'s `PlaceRecommendation` → `PlaceCard.tsx` renders `<RecommendationRing confidence label>` when `recommendation_confidence !== null` (`PlaceCard.tsx:50-54`), else an explicit "Recommendation unavailable" placeholder (lines 63-69) — a deliberate choice so a traveler can distinguish "no score exists" from "still loading."
7. **Pydantic already involved**: yes, both mirrored `PlaceRecommendation` classes (§4) carry `recommendation_confidence: float | None` and `recommendation_label: str | None` as validated fields today.
8. **Must remain unchanged**: the model file, `extract_features()`'s feature order/schema, `predict_recommendation()`'s probability→label logic, the null-when-no-review-text behavior, and the "not an objective quality score" framing that the frontend copy and `PlaceRecommendation`'s field docstring both depend on.

**A genuine existing inconsistency worth flagging** (not a bug, but a real architectural wrinkle): `top_quality_places` (`agent/tools.py:573-606`) ranks by the **old** `ml_predictions.target='quality_score'` value, not by this XGBoost recommendation model. So a request like "find the best-rated cafes" and a request that surfaces the same cafes via `search_places` → `place_details` can show **two different, independently-computed scores** for the same place, without any of the current code reconciling them. This predates the near/far bug entirely and is unrelated to it, but is directly relevant to §6/§19 of the requested investigation (how XGBoost interacts with different candidate-retrieval paths) and to the future architecture's "XGBoost recommendation scoring" stage.

---

## 7. Serper

- **Where used**: `ingestion/web_enrichment.py` (`search_web()`, `filter_results()`) is called from exactly one place, `agent/tools.py`'s `_search_place_evidence()` (line 727).
- **What triggers it**: `_search_place_evidence` is called from two sites, both gated on "this place currently has zero linked review/evidence text":
  1. `_fetch_place_knowledge()` (`agent/tools.py:817-832`) — inside `_lookup_place_structured()`, only when a **curated** place's `reviews_raw` query returns nothing (line 895).
  2. `_discover_live_place()` (`agent/tools.py:472-502`) — for a **brand-new** place resolved via live Nominatim lookup (`search_place_live` tool) that has no curated match at all.
- **What query it receives**: a site-scoped search (`site:{official_domain} {place_name}`) first, using the place's real OSM `website` tag if present, then a general `"{place_name} Copenhagen"` search if that found nothing, then Wikipedia's REST API as a last resort (`agent/tools.py:704-787`).
- **What results it returns / how represented**: filtered by `ingestion.web_enrichment.filter_results` (length/domain screen — not read in this pass, referenced only), then by `_mentions_copenhagen_or_denmark()` (an explicit relevance gate added after a real false-positive incident — a generically-named landmark matched unrelated e-commerce listings, `agent/tools.py:614-628`), then ranked by `_domain_tier()` (official site > known tourism-org allowlist > generic snippet, lines 691-701).
- **Do results enter the database?** Yes — `_store_place_evidence()` (`agent/tools.py:790-814`) inserts them as ordinary `reviews_raw` rows (`source_type='web_search'` or `'wikipedia'`), tagged by tier in `raw_payload`, linked via `place_mentions`. Because a place is only ever enriched **once** (the check is "does `reviews_raw` already have rows for this place," `agent/tools.py:895`), this doubles as a permanent, self-healing cache — no place is ever re-enriched.
- **Do they receive XGBoost scores?** Yes, but **indirectly and unconditionally** — once stored as `reviews_raw` text, `predict_recommendation()` consumes it exactly like any other review text (it's part of `combined_text` in `extract_features()`), producing a real `minilm_good_probability` and `review_count` feature. There is no separate "is this eligible for scoring" branch for Serper-sourced text; it is indistinguishable from seed/Wikivoyage/Reddit text by the time it reaches the model.
- **Do they receive quality scores?** No — the old `quality_score` in `ml_predictions` is a **precomputed, offline batch value** (from `pipeline/modeling/train_quality_model.py`, not read in this pass); a live-discovered place has no row there at all, so `top_quality_places` would simply never surface it (that tool's SQL does an inner `JOIN ml_predictions ... target='quality_score'`, `agent/tools.py:582-583` — a live-discovered place is silently excluded, not scored 0).
- **Duplicates**: no cross-request Serper-call dedup mechanism beyond the "already has review text" gate above; within one request, `@_cache_tool_calls` memoizes identical tool calls.
- **How they reach the final response**: as ordinary review text feeding `predict_recommendation`, and — for genuinely new places — the place row itself (`data_status='live_discovered'`) becomes visible to every other tool (`search_places`, `search_places_near`, etc.) from that point on, indistinguishable from curated data except for the `data_status` column.

**Is this a "proper fallback mechanism" in the sense the user's proposed architecture describes (internal DB insufficient → external candidate discovery)?** **No — this is an important, precise distinction.** Today Serper is *never* a candidate-discovery source for an open-ended request ("find me a cafe near X" always uses `_places_near`/pgvector, never Serper, regardless of how few DB matches exist). Serper only fires as an **evidence-enrichment** step for a single, already-identified named place that has zero review text. The target architecture in §16/the user's own request describes Serper as a *candidate-discovery* fallback ("not enough internal results → Serper finds more candidates") — that is a genuinely new responsibility, not something the current code does in a smaller way.

---

## 8. CrewAI

- **Agents**: Place Scout, Concierge (`agent/crew.py:357-384`) — folded down from an earlier three-agent design; a third "Conditions Analyst" agent was removed and its one job (call weather, summarize) merged into the Concierge, because CrewAI's own fixed per-agent overhead (a fresh opening LLM call plus backstory/task tokens) measured ~1,400-1,600 tokens per run regardless of the agent's actual workload (`agent/crew.py:5-13`).
- **Tasks**: `scout_task`, `concierge_task` (`Process.sequential` — Concierge's task has `context=[scout_task]`, so it sees the Scout's output).
- **Tools**: see §2/§7 above; each is a plain `@tool`-decorated function wrapped in `@_cache_tool_calls` for per-request memoization.
- **Prompts**: the entire natural-language-understanding logic lives in `scout_task.description` and `concierge_task.description` — long, hand-tuned prose blocks (`agent/crew.py:386-449`, `460-517`), not a schema.
- **Model**: `gpt-4o-mini` (`OPENAI_MODEL` env var, default), `temperature=0.3`, `max_tokens=OPENAI_MAX_OUTPUT_TOKENS` (default 1500).
- **Output format**: `output_pydantic=TripPlanOutput` on `concierge_task` only — enforced via `instructor`, not a prompt instruction to "output JSON" (comment, `agent/crew.py:278-280`).
- **Retries**: `max_retries=0` at the OpenAI SDK level (`build_llm()`, line 306) — deliberately disabled so a flaky call can't silently multiply billed attempts; `max_iter=MAX_AGENT_ITER=3` per agent (CrewAI's own reasoning-loop cap). `plan_trip()` itself **never retries** a whole-crew run on failure (comment, lines 1125-1131) — a retry would re-run the entire crew from scratch on a small, fixed API budget.
- **Token limits**: `OPENAI_MAX_OUTPUT_TOKENS=1500` per call (raised from 900 after a real truncated-JSON incident, see §12); `MAX_LLM_CALLS_PER_REQUEST=6` shared across both agents for one trip-plan request, enforced by `_instrument_llm()`'s wrapper around `llm.call` (lines 328-349) — exceeding it raises `TripPlannerLLMUnavailable`, treated identically to a real OpenAI outage.
- **Caching**: two independent layers — (1) `_tool_call_cache` (`agent/tools.py:88`), in-process, reset once per `plan_trip()` call, memoizes identical `(tool, kwargs)` calls within one request; (2) `trip_plan_cache` DB table, 24h TTL, cross-request, keyed on normalized `(request, date, start_location)` or `(request, date)` for the travel-time-only reuse path.
- **Tool call caching**: see (1) above — also repurposed as an introspection mechanism: `get_cached_tool_calls(fn_name)` (`agent/tools.py:95-102`) lets backend code read the Scout's *actual* tool-call arguments after the fact, which is what `_reconcile_near_relationships()` relies on instead of trusting the Concierge's transcription.
- **Deterministic fallback paths**: `deterministic_trip_plan()` (`agent/crew.py:953-1029`, zero LLM cost, used when OpenAI is confirmed unavailable and no reusable cache exists) and `_trip_plan_from_cached_results()` (`agent/crew.py:1032-1121`, reuses whatever tool calls the crew already completed before an LLM failure at the final synthesis step).
- **Decisions currently left to the LLM**: which tool to call per request-part; which place name to pass as `anchor_place`/`query`; whether a relationship is near/far/neutral; which of `search_places`'s returned candidates are genuinely relevant enough to keep (`agent/crew.py:434-439`, an explicit instruction, not code-enforced); all natural-language prose (`why_recommended`, `overall_note`, `weather_summary`).

---

## 9. Distance and relationship system

| Value | Source | Deterministic? |
|---|---|---|
| start location → primary destination (`distance_km`, `walk_minutes`, `bike_minutes`, `travel_note`) | `travel_time_estimate` tool (Concierge-invoked, batched, `agent/tools.py:1050-1094`) using `_trip_start` (set once via `geocode()`) and `haversine_km`/`travel_fields`; backfilled if the Concierge forgets a place via `_ensure_start_distance()` (`agent/crew.py:779-816`) | Yes — math is deterministic; **which places get a start distance at all** originally depended on the Concierge remembering to include them, now backstopped |
| primary → secondary ("near X") | `search_places_near` tool computes real `haversine_km` from the **Scout's own resolved anchor**; then independently **re-derived from scratch** by `_reconcile_near_relationships()` (`agent/crew.py:678-755`) using `get_cached_tool_calls("search_places_near")` — the actual recorded arguments, not the LLM's prose | The distance math is deterministic once a `search_places_near` call exists; **whether that call happens at all, and with which anchor, is still pure LLM judgment** (§3) |
| `near_place` / `near_distance_km` | Set by `_reconcile_near_relationships` from a fresh `_places_near()` query against the real anchor coordinates, `limit=20` (generous, to catch any candidate the Concierge kept even if not the Scout's own top few) | Deterministic |
| `distance_km` (start-relative) is cleared to `None` for any place matched into `near_lookup` | Same function, lines 745-753 | Deterministic — prevents the exact original bug (café showing "6.8 km from your start" instead of "0.3 km from Little Mermaid") |
| walking/biking time, travel note | `travel_fields()` (`agent/tools.py:150-168`) — shared by the live tool and the cache-recompute path | Deterministic |

**Remaining case where the LLM can still produce an incorrect relationship**: the reconciliation layer is a *correction* mechanism, not a *validation* mechanism — it trusts that if `search_places_near` was called, the relationship really should be "near," and if it wasn't called, the relationship really should be "no reference point." It has no way to detect "the Scout called `search_places_near` when it shouldn't have" (the confirmed Test D failure — a bare "then have coffee" request still triggered `search_places_near(anchor_place="the Little Mermaid", ...)`, and the reconciliation layer faithfully treated that as a genuine near-relationship, because from its point of view a real tool call happened with a real, resolvable anchor). **This is the precise architectural gap**: correctness of the *numbers* is fully deterministic; correctness of the *relationship classification itself* is not backed by anything except a single LLM reasoning pass over prose.

---

## 10. Weather

- **Provider**: Open-Meteo (`archive-api.open-meteo.com` for past dates, `api.open-meteo.com` for today/future, `agent/tools.py:1099-1100`), free, keyless.
- **Input**: `target_date` (`YYYY-MM-DD`), optional `category` to scope the Outdoor Interest Index.
- **Date handling**: `_weather_structured()` (`agent/tools.py:1183-1246`) parses the date, checks `weather_daily` cache first, and — on a miss — picks the archive vs. forecast endpoint based on whether the date is in the past; forecast dates beyond Open-Meteo's real 16-day horizon (`_MAX_FORECAST_DAYS`, checked against the live API's own behavior, not assumed) are rejected honestly with `error_kind="out_of_range"` **before** attempting a live call, so that case is never confused with a genuine provider failure.
- **Returned values**: `temp_max_c, temp_min_c, precip_mm, wind_kph`, plus `outdoor_interest` (from `visit_time_forecast`, a *separate*, older, periodically-batch-computed signal — its own date coverage window is independent of Open-Meteo's live 16-day horizon, and the code explicitly does not blur the two, `agent/tools.py:1296-1302`).
- **How it enters the final response**: the Concierge calls `weather_conditions()` once (`agent/crew.py:466-468`) and is instructed to write `weather_summary` from its real output; in the two deterministic (no-LLM) paths, `_weather_summary_text()` (`agent/crew.py:819-833`) formats the same structured data directly, with **distinct, honest messages** for `unparsable_date`, `out_of_range`, `rate_limited`, and `provider_unavailable` (`agent/tools.py:1249-1265`) — a real prior bug (a genuine Open-Meteo 429 being reported identically to "beyond forecast horizon") is what forced this split, see §12.
- **Where the weather text is generated**: either the Concierge's own prose (live path) or `_weather_summary_text()` (deterministic paths) — both ultimately describe the same structured dict, never inventing numbers.
- **Caching**: `weather_daily` table, upserted on every live fetch (`ON CONFLICT (date) DO UPDATE`, `agent/tools.py:1170-1179`) — permanent cache per date, not time-limited (weather for a past date never changes; a future date's forecast is only fetched once and then treated as settled, which is a real, if minor, staleness trade-off not otherwise flagged in the code).

---

## 11. Frontend

Relevant Trip Planner components, all in `web/components/trip-planner/`:

```
TripPlannerForm (input: request text, area, target date, start location)
        │ onSubmit
        ▼
TripPlannerClient ("use client")
        │ planTrip() — POST /trip-plan, AbortController for cancel
        ▼
   status: idle | loading | error
        │
   loading → LoadingStatus (with cancel button)
   error   → inline error box (backend's HTTPException detail shown as-is)
   success → TripPlanResults
                 ├─ ConditionsBox (weather_summary)
                 ├─ PlaceCard × N
                 │    ├─ RecommendationRing (if recommendation_confidence != null)
                 │      else "Recommendation unavailable" placeholder
                 │    ├─ meta line: vibe_cluster · opening_hours (only if present)
                 │    ├─ travel box: near_place/near_distance_km PRIORITIZED over
                 │    │   distance_km (PlaceCard.tsx:19-34) — already correct
                 │    ├─ why_recommended
                 │    ├─ summary (if present)
                 │    └─ sources (if present)
                 └─ overall_note (if present)
```

`web/lib/types.ts` mirrors `api/main.py`'s Pydantic models field-for-field; `web/lib/api.ts` is a plain `fetch` wrapper, deliberately not proxied through a Next.js API route because of the 1-2 minute request duration (§2). **No UI changes are required by anything in this report** — the display logic already correctly branches on `near_place`/`near_distance_km` vs. `distance_km`, and this was independently confirmed by reading the file (not just inferred).

---

## 12. Existing bugs and fixes — chronological

This list covers the real, verified bug history for the near/far relationship problem specifically (the subject of this investigation), reconstructed from the current session plus `git log`. Commit-message-only entries (not independently re-verified against a diff in this pass) are marked as such.

1. **Original bug**: *"see the Little Mermaid and have coffee afterwards nearby"* showed the café's distance as "6.8 km from your start" instead of its real distance from the Little Mermaid.
   - **Root cause**: `travel_time_estimate` unconditionally measures every place from the trip's start (`agent/tools.py:1050-1094`); the Concierge separately had to correctly transcribe the Scout's own near-distance text into `near_place`/`near_distance_km` — a fragile LLM-to-LLM hand-off with no verification.
   - **Fix**: `_reconcile_near_relationships()` (`agent/crew.py:678`) — stop trusting any LLM-transcribed number; re-derive the real anchor and real haversine distance from the Scout's actual recorded `search_places_near` tool-call arguments (`get_cached_tool_calls`), independently, from the database.
   - **Tests**: `test_reconcile_near_relationships_fixes_a_wrong_reference_distance` and others in `tests/test_crew.py` (lines 295-397). Fully solved for the case where `search_places_near` was correctly called.

2. **Primary place missing its own start distance**: the Concierge's batched `travel_time_estimate` call sometimes omitted the primary place's own name (self-discovered during real-call verification, not user-reported).
   - **Root cause**: no code guarantee the Concierge asks for every place, only an instruction.
   - **Fix**: `_ensure_start_distance()` (`agent/crew.py:779`) — deterministic backfill using `get_trip_start()`'s already-geocoded coordinates, only filling genuinely missing values.
   - **Tests**: `test_ensure_start_distance_*` (`tests/test_crew.py:398-437`). Solved.

3. **Shortened place name mismatch**: the Concierge sometimes wrote "Terminalen kaffebar" for the DB's real "Terminalen kaffebar - Seaside Toldboden," causing an exact-match lookup to silently miss the correction.
   - **Fix**: `_find_near_match()`'s substring fallback (`agent/crew.py:758-776`). Solved.

4. **The "far away" regression** (introduced by this session's own Round-1 prompt edit, then fixed in Round 3): treating "then"/"afterwards" as a positive proximity signal caused *"see the Little Mermaid and then go to a cafe far away"* to still call `search_places_near(anchor_place="Den lille Havfrue", ...)`.
   - **Fix**: restructured `scout_task`'s location-relationship instructions into an explicitly-ordered 3-case priority (far/distant wins > explicit proximity > neither — `agent/crew.py:398-433`, current text).
   - **Real end-to-end verification this session**: an isolated, clean re-run of the "far away" phrasing (fresh date, single-process log capture) confirmed `Tool: search_places` was called for the secondary part (not `search_places_near`), and the returned cafés (Café G, Cafelitten, Bastard Cafe) all had `near_place=None` and plausible start-relative distances (5.0-5.4 km) — consistent with the fix working for this case.

5. **Test D — CONFIRMED, UNRESOLVED**: a bare *"I want to see the Little Mermaid and then have coffee."* (no explicit spatial word at all) still produced `search_places_near(anchor_place="the Little Mermaid", category="cafe", ...)` in a real end-to-end call, directly contradicting the prompt's own explicit non-triggering example. This is the concrete evidence that a prose-only, single-LLM-pass decision boundary is not reliable regardless of how explicitly it's worded — this is the reason the current session was redirected toward this architecture investigation instead of another prompt edit. **Not fixed. No further prompt change has been attempted since this was found**, per explicit instruction to stop patching and investigate structurally instead.

6. **Compound-request primary-place loss** (git `0c1b93a`, commit-message only): the deterministic (no-LLM) fallback path used to embed a whole compound sentence as one semantic query, letting the secondary category's words dilute the primary landmark out of the results. Fixed by `_split_compound_request()`/`_compound_deterministic_places()` (`agent/crew.py:844-950`).

7. **Groq → OpenAI migration** (git `0b7d12b`, commit-message only, consistent with code comments in `agent/crew.py:15-23`): Groq's free-tier 12,000 TPM ceiling proved unreliable in production; switched to `gpt-4o-mini`.

8. **Truncated `weather_summary`** (git `db76ac0`): `OPENAI_MAX_OUTPUT_TOKENS=900` was observed live to cut off mid-sentence inside `weather_summary` on a real multi-place request — `instructor`'s retry-on-invalid-JSON closed the syntax but never rewrote the truncated sentence, so the response looked "successful" while silently shipping a broken sentence. Fixed by raising to 1500 (`agent/crew.py:99-124`).

9. **Missing Serper key crash** (referenced in `agent/tools.py:729-736`, git `4192d3d`): `search_web()` read `SERPER_API_KEY` directly and raised a bare `KeyError` when unset, propagating uncaught as an HTTP 500 (frontend only showed "Failed to fetch"). Fixed by checking `has_serper_key` first and skipping straight to the keyless Wikipedia fallback.

10. **Render OOM** (git `c614082`, `aeaab65`): confirmed real memory crash (exit 137) from `sentence-transformers`'/`sklearn`'s `torch` import; fixed by switching to `fastembed` (ONNX) and the native `xgboost.Booster` API, dropping `scikit-learn` and `torch` from the live runtime path entirely (kept only under a separate `modeling` extra for offline training).

11. **Weather rate-limit/out-of-range conflation** (referenced in `agent/tools.py:1118-1124`, `1249-1265`): a real Open-Meteo 429 was once reported to a traveler identically to "beyond the forecast horizon" for a date only 8 days out. Fixed by splitting `error_kind` into `rate_limited`/`provider_unavailable`/`out_of_range`/`unparsable_date`.

---

## 13. Current test status

- **Test files**: `tests/test_ab_scoring.py`, `test_agent_tools.py`, `test_api.py`, `test_crew.py`, `test_neighborhood_backfill.py`, `test_rag_summary.py`, `test_ranking.py`, `test_recommendation_service.py`, `test_schema.py`, `test_validation.py`, `test_web_enrichment.py`.
- **Full suite run this session**: `test_neighborhood_backfill.py` **fails to collect** — `ModuleNotFoundError: No module named 'shapely'` (`ingestion/neighborhood_backfill.py:37`). This is a local Python-environment gap (missing dependency), unrelated to the Trip Planner or this investigation, and was not touched.
- **Everything else**: `106 passed` in one run (`pytest tests/ --ignore=tests/test_neighborhood_backfill.py`, 79.84s). All currently pass together, including all 23 tests in `tests/test_agent_tools.py` and all 23 in `tests/test_crew.py` covering the near/far reconciliation logic specifically.
- **Note on a previously-reported flaky test**: an earlier session noted `test_trip_plan_falls_back_to_deterministic_results_on_an_openai_failure` failing only when run *after* `test_crew.py` in the same process, and diagnosed it as pre-existing (reproducible on unmodified `HEAD` via `git stash`), not something this session's changes caused. **This was not re-verified in this pass** — the full suite ran clean once here — flagging it as "previously observed, not reproduced this run" rather than claiming it's fixed.
- **What has actually been verified with a real LLM request** (not just reasoned about — this distinction matters given the user's explicit instruction): four real, live, end-to-end `POST /trip-plan` calls via `fastapi.testclient.TestClient` with a real `OPENAI_API_KEY`, fresh dates to bypass caching, this session:
  - "...have coffee afterwards nearby" — **PASSED**: `search_places_near(anchor_place="Den lille Havfrue", ...)` called, all cafés got real `near_place`/`near_distance_km` (0.32/0.56/0.59 km), `distance_km=None` for them.
  - "...find a cafe within 1 km afterwards" — **PASSED**: same structure, `max_km=1` respected.
  - "...then go to a cafe far away" — **PASSED** (isolated re-run, direct log confirmation): `search_places` used for the secondary part; three different cafés, all `near_place=None`, start-relative distances 5.0-5.4 km.
  - "...and then have coffee" (bare, no spatial word) — **FAILED**, confirmed by direct log inspection: `search_places_near(anchor_place="the Little Mermaid", category="cafe", ...)` was called, contradicting the prompt's own explicit rule and worked example.
- **What was only reasoned about, not verified**: the general claim that the reconciliation/backfill functions are "correct for any place/category" beyond the specific unit-test and real-call cases exercised above — the unit tests are synthetic/deterministic and don't exercise a live LLM decision.

---

## 14. Current Git state

- **Branch**: `main`, up to date with `origin/main` (`07d2259`).
- **A second local branch exists**: `backup-before-trailer-strip` (`375e511`) — not investigated further, presumed an intentional prior safety branch.
- **Uncommitted changes** (working tree, not staged): `agent/crew.py` (+238/-41 relative to HEAD, roughly), `agent/tools.py` (+77 lines), `tests/test_crew.py` (+175 lines) — this is the entire near/far relationship fix built across this session (the `_reconcile_near_relationships`/`_ensure_start_distance`/`_find_near_match` functions, the `max_km`/`_coerce_max_km` support, the restructured `scout_task` prose, and their unit tests). **None of this has been committed or pushed**, consistent with every explicit instruction given this session.
- **Recent real commits on `main`** (see §12 for which ones were cross-referenced against code vs. taken from the message alone): `07d2259` copy polish, `03114fe` copy polish, `db76ac0` weather truncation fix, `0b7d12b` Groq→OpenAI, `8ab8890` Groq fallback widening, `01b850d` scaffold cleanup, `4192d3d` Serper key fix, `0c1b93a` compound-request fallback fix, `c614082`/`aeaab65` OOM fixes.
- **Does everything currently run from latest pushed code?** No — the working tree has real, functional, uncommitted changes (above) that are not on `origin/main`. Anyone pulling `main` fresh right now would get the codebase **without** the near/far reconciliation system described in §9/§12 items 1-5.

---

## 15. Architecture weaknesses

Distinguishing LLM responsibility from backend responsibility, honestly, across each dimension the user asked about:

| Dimension | Today | Backed by code, or by prose alone? |
|---|---|---|
| **Which tool to call** (search_places vs. search_places_near vs. top_quality_places vs. search_place_live) | Entirely LLM judgment, one reasoning pass | **Prose only.** No structural check exists that a chosen tool matches what the request actually said. |
| **Spatial relationship** (near/far/neutral) | Entirely LLM judgment, from the same reasoning pass that chooses the tool | **Prose only** — confirmed unreliable by Test D (§12 item 5, §13). |
| **Ordering/sequence** ("then", "before", "first") | Not modeled as a distinct concept anywhere — conflated with tool choice in the same prose block, which is precisely how the "far away" regression happened (a sequence word was briefly treated as a proximity signal) | **Prose only**, and the current prose only distinguishes it well enough to avoid one specific failure mode (sequence-as-proximity); "before" specifically is never mentioned in the rules at all (§3 table). |
| **Distance/relationship numbers, once a tool call happened** | Real haversine math, independently re-derived from the DB using the actual recorded tool-call arguments | **Deterministic backend.** This part of the architecture is already sound. |
| **Candidate selection** (which `search_places` hits are kept) | LLM judgment ("use your own judgment on relevance, not just whatever the tool ranked highest," `agent/crew.py:434-439`) | **Prose only**, though `api/ranking.py`'s deterministic relevance floor/gap already discards the worst tail before the LLM ever sees it — a real, if partial, backend contribution here. |
| **Structured intent** | Does not exist as a concept in the codebase | N/A — this is the literal gap the user's proposed architecture fills. |
| **Neighborhood constraints** ("around Nørrebro") | Not modeled — `_search_places_rows` has a `neighborhood` parameter but nothing in the current Scout prompt tells it to populate it from free text | **Not implemented at all**, not merely fragile. |
| **XGBoost / quality-score duality** (§6) | `top_quality_places` uses a different, older, less-validated score than `place_details`/every other path | Not a near/far issue, but a real, separate inconsistency worth the reviewer's attention. |
| **Serper's actual role** (§7) | Evidence enrichment for one already-identified place, never candidate discovery | The target architecture's "internal insufficient → Serper" flow does not exist today in any form — this would be new functionality, not a refactor of existing logic. |

**Overall assessment**: the backend is genuinely strong at *executing* a relationship once it is known — every distance number in the system is real, independently-computed, and defensible. The entire, sole point of fragility is *deciding what the relationship is* from free text, which today happens in exactly one place (a single LLM reasoning pass over an ever-growing block of prose rules) with zero structural backstop. This matches the user's own diagnosis precisely.

---

## 16. Proposed future architecture (not implemented — investigation only)

The user's proposed flow:

```
USER NATURAL LANGUAGE
        ↓
LLM / CrewAI INTENT EXTRACTION
        ↓
STRICT PYDANTIC ITINERARY SPECIFICATION
        ↓
BACKEND VALIDATION
        ↓
POSTGRESQL / PGVECTOR SEARCH
        ↓
ENOUGH INTERNAL RESULTS?
   ↓ YES → CONTINUE          ↓ NO → SERPER FALLBACK
        ↓
VALIDATE / NORMALIZE RESULTS
        ↓
XGBOOST RECOMMENDATION SCORE
        ↓
DETERMINISTIC DISTANCE / RELATIONSHIP CALCULATION
        ↓
WEATHER
        ↓
FINAL PYDANTIC RESPONSE
        ↓
FRONTEND
```

Is this technically appropriate given what exists today? Stage by stage:

- **LLM/CrewAI intent extraction**: technically sound and a natural extension of the existing Scout agent — instead of the Scout directly choosing tools, it would produce a structured object (e.g. `primary: PlaceRef | None`, `secondary: list[ItineraryPart]` where each part has `category_or_query: str, relation: Literal["near","far","neutral"], anchor: str | None, max_km: float | None, min_km: float | None, neighborhood: str | None, sequence: Literal["before","after",None]`). This can reuse `output_pydantic`/`instructor` exactly as `concierge_task` already does today (§8) — the mechanism already exists in this codebase for a different task, so this is not a new dependency, just applying an existing, working pattern one step earlier in the pipeline. `agent/crew.py`'s current `PlaceRecommendation`/`TripPlanOutput` validators (null-coercion, placeholder-text repair, §4) demonstrate the team already has a working pattern for handling imperfect LLM structured output; the same pattern would apply to a new `TripSpecification` model.
- **Backend validation**: new — a place for e.g. `_coerce_max_km`-style repair to live for the new structured fields, and for cross-field checks (e.g. `relation="far"` should imply `anchor is None` allowed, `max_km is None` when `relation != "near"`).
- **PostgreSQL/pgvector search**: unchanged — `_search_places_rows`, `_places_near`, `_resolve_place` already exist and already do exactly the deterministic work this stage needs; they would be called directly from backend code driven by the structured spec's `relation` field, instead of being *offered as tool choices* to the LLM. This removes the LLM from the decision of *which* search function to call — it only ever supplies *what* to search for and *what the relationship is*, and the backend maps `relation` deterministically to a specific already-existing function.
- **"Enough internal results?" → Serper fallback**: **new functionality**, not present today (§7's finding). Would need a defined, explicit threshold (e.g. `_search_places_rows` returning fewer than N results after `api/ranking.py`'s relevance filter) and a new candidate-shaped Serper query path distinct from the existing evidence-enrichment one — these are different concerns (find new places vs. find text about an already-identified place) that would need to stay separate, not merged.
- **Validate/normalize/dedupe external results**: new — would need to reuse `_map_nominatim_category`-style category mapping and the existing `_domain_tier`/`_mentions_copenhagen_or_denmark` relevance gates (§7), which already exist for the live-discovery path and are directly reusable.
- **XGBoost recommendation scoring**: unchanged — `predict_recommendation()` already runs unconditionally on any place with review text, curated or live-discovered (§6, §7) — no new integration work needed here, this stage already exists exactly as described.
- **Deterministic distance/relationship calculation**: largely unchanged — `haversine_km`, `travel_fields`, `_places_near` already do this; what changes is that the structured spec's `relation` field becomes the trigger, replacing today's `_reconcile_near_relationships`-after-the-fact-correction pattern with an upfront, backend-driven decision. This is a genuine simplification, not just an addition: today's reconciliation function exists specifically to *undo* a wrong LLM tool choice after the fact; with a structured spec validated before any DB search runs, there may be nothing left to reconcile.
- **Weather**: entirely unchanged — already a clean, isolated, well-tested subsystem (§10).
- **Final Pydantic response**: unchanged in spirit — `TripPlanOutput`/`api/main.py`'s `TripPlanResponse` already do this.
- **Frontend**: unchanged — confirmed in §2/§11, the display layer already correctly prioritizes `near_place`/`near_distance_km`.

**Conclusion on appropriateness**: yes, technically sound, and notably it is **less** of a rewrite than it might first appear — most of the deterministic backend machinery this architecture needs (search, distance math, category mapping, relevance filtering, XGBoost scoring, weather) already exists and already works; the actual net-new work is (a) a `TripSpecification` Pydantic model, (b) moving the Scout's decision from "pick a tool" to "fill in a schema," (c) backend code that maps the schema deterministically onto the *existing* search functions instead of the LLM picking among them, and (d) a genuinely new Serper candidate-discovery path with an explicit "insufficient results" threshold. Item (c) in particular would likely let `_reconcile_near_relationships()`'s after-the-fact correction logic be simplified or removed, since the relationship would be known and validated *before* any search runs, not inferred and corrected afterward.

---

## 17. What should NOT change

Explicitly, per direct inspection:

- **XGBoost**: `agent/recommendation_service.py` — model file, `extract_features()`, `predict_recommendation()`, native-Booster loading approach (never switch back to `XGBClassifier`, which reintroduces the `scikit-learn`/OOM problem).
- **Existing recommendation_confidence/recommendation_label scoring and framing**: the "not an objective quality score" distinction, the null-when-no-review-text behavior.
- **pgvector / PostgreSQL**: `db/schema.sql`, the HNSW index, `_search_places_rows`, `_places_near`, `_resolve_place` — these are the exact functions the proposed architecture would call directly, not replace.
- **Existing recommendation/quality-score features**: `extract_features()`'s feature set and order (`recommendation_feature_schema.json`).
- **Weather**: `agent/tools.py`'s `_weather_structured`/`_fetch_and_cache_live_weather`/`weather_conditions`, the `weather_daily` cache, the distinct `error_kind` handling.
- **Frontend design**: `PlaceCard.tsx`, `RecommendationRing.tsx`, `TripPlanResults.tsx` — confirmed already correct for the near/far display problem; no changes identified as necessary anywhere in this investigation.
- **Existing place data**: `places`, `reviews_raw`, `ai_summaries`, etc. — no schema change identified as required; a new `TripSpecification` concept is application-layer, not a new table (though a `neighborhood`-constraint field, if added, would just exercise `_search_places_rows`'s already-existing but currently-unused `neighborhood` parameter — no schema change).
- **Existing APIs**: `/trip-plan`'s request/response shape (`TripPlanRequest`/`TripPlanResponse`) — the proposed architecture is an internal pipeline change; nothing about it requires changing what the frontend sends or receives.

---

## 18. Final recommendation

**A. What is already working well:**
- The deterministic distance/relationship *math* (haversine, travel time, radius capping) — real, tested, correct.
- The post-hoc reconciliation pattern (`_reconcile_near_relationships`, `_ensure_start_distance`) — a genuinely good idea, well-tested, that fixed four real distinct bugs (§12 items 1-3).
- XGBoost recommendation scoring — correctly isolated from the live process's memory constraints, correctly framed, correctly null-safe.
- Weather — clean, well-tested, honest about its own failure modes.
- The frontend — already correctly built for exactly the display logic this whole investigation concerns; genuinely requires zero changes.
- Caching (both in-request tool-call memoization and cross-request `trip_plan_cache`) — real cost/latency wins with no correctness compromise found.

**B. What is currently fragile:**
- The single point where natural language becomes a decision about *which* tool to call and *what relationship* it implies (§3, §9, §15) — confirmed, reproducibly, to fail on wording the current prompt explicitly and unambiguously covers (Test D).
- The `top_quality_places`/live-recommendation-model duality (§6) — not related to near/far, but a real inconsistency for a future reviewer to be aware of.
- Serper's narrow role (§7) — not itself broken, but far short of the "candidate discovery fallback" the user wants, which doesn't exist in any form yet.
- Neighborhood constraints ("around Nørrebro") and "before" as a sequence word — not fragile, simply **absent** from the current rules entirely (§3).

**C. What should be changed** (pending the user's review of this report, not started):
- Add a structured intent layer (§16) so relationship classification is validated *before* a DB search runs, not corrected *after* an LLM's ad hoc tool choice.
- Give Serper a genuine second role as a candidate-discovery fallback, kept structurally distinct from its existing evidence-enrichment role.
- Decide (a separate, smaller decision) whether to reconcile the `quality_score`/recommendation-confidence duality, or leave `top_quality_places` as intentionally distinct.

**D. What should NOT be changed:** everything in §17.

**E. The smallest safe implementation plan** (proposed, not started):
1. Define `TripSpecification` (and part-level `ItineraryPart`) as a new Pydantic model, reusing the existing `field_validator`-repair pattern already proven on `PlaceRecommendation`.
2. Add a new CrewAI task (or repurpose `scout_task`) with `output_pydantic=TripSpecification`, reusing the exact `instructor`-enforcement mechanism already used for `concierge_task` today — no new library, no new pattern.
3. Add backend code that maps a validated `TripSpecification` directly onto the *existing* `_search_places_rows`/`_places_near`/`_resolve_place` functions — this is new glue code, not new search logic.
4. Only once step 3 is proven correct against real requests, consider retiring `_reconcile_near_relationships()`'s after-the-fact correction — not before, since it remains a valid safety net during the transition.
5. Add the Serper "insufficient internal results" branch as new, clearly-separate code from `_search_place_evidence` (which stays exactly as-is for its existing enrichment role).
6. Re-run the exact four real end-to-end tests from §13 (A/B/C/D) plus the "around Nørrebro"/"before"/"near me" cases from §3 that are currently unhandled, as the acceptance bar — not just the existing 106 unit tests, which don't exercise real LLM decision-making.

**F. Risks:**
- A structured-intent LLM call is still an LLM call — it can still misclassify a relationship; the win is that misclassification becomes *visible and testable in isolation* (a wrong field in a validated object) rather than *entangled with tool execution* (a wrong tool call that already happened and touched the database/API budget).
- Adding a Serper candidate-discovery path is new, unexercised code — needs the same honesty-over-confidence discipline already used everywhere else in this codebase (say "not enough results" rather than padding with weak matches, matching `api/ranking.py`'s existing philosophy).
- Any schema change to the request/response contract must be re-verified against the frontend, even though §11 found zero required changes for the current problem — a *new* field (e.g. surfacing `neighborhood` constraints) would need a corresponding, currently-nonexistent frontend affordance if it's ever meant to be shown, not just computed.

**G. Tests required before any deployment of a future change:**
- All existing 106 tests continue passing (baseline, already true today).
- New unit tests for `TripSpecification` validation/repair, mirroring the existing `field_validator` test style in `tests/test_crew.py`.
- Real end-to-end LLM verification (not just unit tests) for at minimum: explicit near, explicit far, explicit distance limit, bare sequence word (no spatial word), "before" as a sequence word, a neighborhood-only constraint, and "near me" — because this session's own history (§12, §13) demonstrates that unit tests over deterministic code do not catch LLM decision-boundary failures; only real calls do.
