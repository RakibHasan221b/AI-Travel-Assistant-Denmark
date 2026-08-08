# Project Status — AI Denmark Explorer

Snapshot as of 2026-08-09. Written after the Trip Planner fallback fix
was verified (focused tests + full suite + real browser run) so a future
session can pick up context without replaying this one. This document
records what exists and what was verified — it does not propose changes.

## 1. Current architecture

**Explore retrieval and ranking** — `GET /explore` (`api/main.py`) embeds
the query with the shared FastEmbed singleton, does a pgvector cosine
similarity search (`ORDER BY embedding <=> qvec`) against a generous
candidate pool, then reranks with `api/ranking.py`'s
`rank_explore_candidates()`: a deterministic combination of the raw
`similarity`, a lexical `name_match_score` (exact/substring/token-overlap
against the place's own name), and a `category_intent_score` (small
keyword→category map). Candidates below `RELEVANCE_FLOOR` or too far
behind the best `combined` score (`RELEVANCE_GAP`) are dropped. This
exists because raw pgvector similarity alone padded results with
loosely-related places.

**Trip Planner normal path** — `plan_trip()` (`agent/crew.py`) runs a
3-agent CrewAI crew against Groq (`llama-3.3-70b-versatile`) with tools
from `agent/tools.py` (`search_places`, `search_places_near`,
`top_quality_places`, `place_details`, `search_place_live`). Tool calls
are cached per-request in `_tool_call_cache` (keyed by function name +
sorted kwargs) so repeated lookups within one crew run don't hit the DB
twice.

**Trip Planner deterministic fallback** — used when Groq is unavailable
(rate limit / quota / provider error). Two paths, tried in order by
`api/main.py`'s exception handler:
1. `_trip_plan_from_cached_results()` — if any tool call already
   succeeded before Groq failed (checked via
   `agent.tools.get_cached_tool_calls`), reuse that real result instead
   of a fresh search.
2. `deterministic_trip_plan()` — otherwise, run a fresh deterministic
   (non-LLM) search. This is where compound-request splitting lives (see
   §4).

**Fresh place discovery** — `search_place_live()` in `agent/tools.py`,
a last-resort tool used only after the curated-DB tools find nothing.
See §3.

**Tier 1 (curated identity resolution)** — `_find_curated_match()`:
before ever creating a new place, checks whether the Nominatim result is
actually an existing curated place under different wording, first by
exact `osm_id` match, then by coordinate proximity (`_COORD_MATCH_RADIUS_KM
= 0.05` km, ~50m, chosen to absorb geocoding jitter without matching a
genuinely different nearby place).

**Tier 2 (genuinely new place discovery)** — `_discover_live_place()`:
only reached if Tier 1 found no match. Gathers real evidence first (see
below), and only inserts a `places` row if evidence was actually found —
never persists a bare Nominatim hit with no evidence.

**Serper evidence collection** — `_search_place_evidence()` (Wikipedia /
Wikivoyage first, Serper as fallback) fetches real web text for a place
name; results are stored via `_store_place_evidence()`.

**ML recommendation scoring** — `agent/recommendation_service.py`'s
`predict_recommendation()`. Combines a precomputed DistilBERT sentiment
score (`ml_predictions`, target=`distilbert_sentiment`) with a live
MiniLM-based signal computed from current review text, feeding both into
a trained XGBoost model (`pipeline/modeling/rating_model.json`). This is
the model behind `recommendation_confidence` — see §2 for why it's
distinct from the older `quality_score`.

**Frontend recommendation ring** — `RecommendationRing.tsx` renders
`recommendation_confidence` as a colored ring (green ≥80 "recommended",
amber ≥60 "consider", red below 60), driven verbatim by the backend
value — no recomputation in the frontend. `PlaceCard.tsx` shows an
explicit "Recommendation unavailable" label (not a ring, not "0%") when
`recommendation_confidence` is `null`, since null and zero mean
different things (no score exists vs. a genuinely bad score).

## 2. Score definitions

These four numbers look similar but come from different places and mean
different things. Do not conflate them.

| Name | Where it's from | What it means |
|---|---|---|
| **Explore similarity / match score** | `api/ranking.py`'s `combined` value: raw pgvector cosine `similarity` + weighted `name_match_score` + `category_intent_score` | How well an Explore search result matches the *query text*. Purely deterministic string/vector math — no ML model, no review data involved. |
| **Explore `quality_score`** | `ml_predictions` table, `target = 'quality_score'`, joined into `/explore` and `/stats` responses | An older, precomputed-offline quality prediction for curated places. Still surfaced in Explore/Stats. Empirically weaker (r=0.171) than the newer recommendation model below (r=0.668) — kept only for backward display, not used by Trip Planner. |
| **Trip Planner `recommendation_confidence`** | `agent/recommendation_service.py`'s `predict_recommendation()`, an XGBoost model combining precomputed DistilBERT sentiment + live MiniLM review-text signal | **A probability from a trained recommendation model, expressed as 0-100%** — not a generic review score, not the same thing as `quality_score`. `null` when there's no review text or the model itself fails to load (distinct reasons, both surfaced honestly via `confidence_unavailable_reason`). |
| **`recommendation_label`** | Same `predict_recommendation()` call, `recommendation["label"]` | The categorical label paired with `recommendation_confidence` (e.g. "recommended"). `null` whenever `recommendation_confidence` is `null`. |

## 3. Fresh place pipeline

```
Nominatim search
  → Tier 1: curated identity check (_find_curated_match)
      exact osm_id match, else ~50m coordinate match
      → if matched: tell caller to use the existing curated place, no new row
  → Tier 2: genuinely new place (_discover_live_place)
      gather real evidence FIRST (Wikipedia/Wikivoyage, then Serper)
      → if no evidence found: return None, caller reports "insufficient
        evidence" honestly — no row is written, no score is invented
      → if evidence found: INSERT the places row with
        data_status='live_discovered', source_url = first evidence link,
        then run the same predict_recommendation() ML pipeline as any
        curated place
```

- **Exact osm_id matching**: `WHERE osm_id = %s`, checked first since
  it's a perfect identity match when Nominatim returns a clean OSM type/id.
- **Coordinate matching**: haversine distance ≤ 0.05km fallback for
  results without a clean OSM id.
- **`live_discovered` provenance**: `places.data_status` (`curated` |
  `live_discovered`, DB-enforced via `CHECK`) plus `source_url` distinguish
  live-found rows from the original curated OSM ingestion at query time —
  no separate ingestion-log table.
- **Insufficient evidence behavior**: no DB write happens at all;
  `search_place_live()` returns an honest "no evidence found, treat as
  unscored" message. No invented confidence is ever shown.
- **Duplicate prevention**: Tier 1's identity check runs unconditionally
  before Tier 2, so a place already in the curated set (even under
  different wording) is never re-inserted as a second row.

## 4. Trip Planner fallback fix

**The bug**: when Groq failed mid-request on a compound request (e.g.
"see the little mermaid, then find a nearby restaurant"), the old
`deterministic_trip_plan()` treated the whole sentence as one semantic
search query. That search's top matches were weighted toward the generic
half of the sentence ("restaurant"), and — critically — any tool-call
work Groq's crew had already completed (like a real `place_details` call
for the primary landmark) was discarded outright and re-searched from
scratch. The net effect: the named landmark the user explicitly asked
for could vanish from the results entirely.

**The fix**:
1. `_trip_plan_from_cached_results()` — if a tool call already succeeded
   before Groq failed (checked via the request-scoped `_tool_call_cache`),
   that real, already-scored result is reused instead of running any new
   search.
2. `_split_compound_request()` — a regex-based split (`_COMPOUND_SPLIT_RE`,
   `_NEAR_RE`, reusing `api.ranking`'s `_CATEGORY_KEYWORDS`/`normalize_text`)
   separates a compound sentence into a **primary** query (the named
   place) and a **secondary** query (a category, e.g. "restaurant"),
   only when a "near/nearby" relationship is actually present. A plain
   single-intent request returns `None` and falls through to the old
   whole-sentence path unchanged.
3. `_compound_deterministic_places()` — resolves the primary place first,
   then ranks secondary candidates by **real haversine distance from the
   primary place's own coordinates** — never from the user's own
   `start_location`, never by semantic similarity. This distance is kept
   in `near_place`/`near_distance_km`, distinct from `distance_km`/
   `walk_minutes`/`bike_minutes` (which stay relative to the traveler's
   own start location).

**Exact example used throughout testing**: *"wanna see little mermaid
and after go to a nearby restaurant"* → primary = Den lille Havfrue,
secondary = nearby restaurants ranked by real distance from Den lille
Havfrue (not from the user's own starting point).

## 5. Verified results

Only what was actually run and observed:

- Explore "little mermaid" returned the relevant Little Mermaid result.
- Explore "coffee" returned relevant cafes.
- Fresh place Tier 1 (curated identity resolution) worked — a Nominatim
  hit for the Little Mermaid matched the curated row by `osm_id`.
- Fresh place Tier 2 (genuinely new place discovery) worked — a place
  with no curated match and real evidence was persisted as
  `live_discovered` with a real `recommendation_confidence`.
- Insufficient evidence did not fabricate a score — verified the
  no-evidence path returns `None`/an honest message, no row written.
- Duplicate discovery was prevented — a second lookup of an
  already-curated place did not create a second row.
- Trip Planner fallback was tested in a real browser (not mocked) against
  a live local stack, with Groq genuinely unavailable (real quota
  exhaustion, not simulated):
  - Little Mermaid appeared as the primary place.
  - Recommendation confidence was 97%.
  - The recommendation ring rendered green.
  - Nearby restaurants (3 real curated restaurants) were genuinely near
    the Little Mermaid (~0.30km each), not near the user's start point.
  - Weather was present.
  - Travel information (3.12km / 37min walk / 12min bike from Copenhagen
    Central Station) was present and correctly kept distinct from the
    "near" distances.
- 96 tests passed. One collection error
  (`tests/test_neighborhood_backfill.py`, missing `shapely`) is a known,
  pre-existing environment gap — `shapely` belongs to the separate
  `ingestion` extra, not the `agent` extra installed in this venv — and
  is unrelated to any of the above.

## 6. Known limitations

- The late-Groq-failure cache-reuse path
  (`_trip_plan_from_cached_results()`) has direct focused test coverage,
  but was not forced through a live browser run — the real browser test
  happened to hit the earlier failure point (`deterministic_trip_plan()`'s
  compound split) because Groq failed before any tool call had cached
  anything that run.
- Fresh (`live_discovered`) places have incomplete structured metadata
  compared with curated places (no opening hours, no subcategory, etc. —
  only what Nominatim + evidence search actually returned).
- Live DistilBERT inference is not used — a real memory measurement found
  loading the transformers DistilBERT pipeline too costly for the
  runtime process, so DistilBERT sentiment is precomputed offline and
  stored in `ml_predictions`. When a place has no precomputed score, the
  existing degradation path treats it as absent (`has_distilbert_score =
  False`, contributes 0 rather than crashing) — this is existing,
  intentional behavior, not a gap introduced by this fix.
- Groq availability can still force the deterministic fallback for any
  request, compound or not — the fallback is now more correct, not a
  guarantee Groq stays up.
- Nominatim can ambiguously resolve vague queries (e.g. generic terms
  with multiple real matches); the pipeline takes its top result as-is.

## 7. Current uncommitted work

`git status` (unstaged, nothing committed):

**From the Trip Planner fallback fix (this session and the one before it):**
- `agent/crew.py` — compound-split fallback logic
- `agent/tools.py` — `get_cached_tool_calls()`, `_places_near()` factor-out
- `api/main.py` — tries cached-result reuse before the deterministic fallback
- `tests/test_crew.py` — the 4 new fallback regression tests

**Earlier, pre-existing uncommitted work (predates this fix, not touched
by it — from the fresh-place-discovery / recommendation-model phase):**
- `agent/tools.py` *(same file — also carries the Tier 1/2 discovery
  logic and `place_details()` rewrite from that earlier phase)*
- `agent/crew.py` *(same file — also carries the earlier phase's
  `_place_recommendation_kwargs()`/`_weather_summary_text()` groundwork)*
- `db/schema.sql` — `data_status`/`source_url` columns
- `tests/test_agent_tools.py` — coverage for the discovery pipeline
- `web/components/trip-planner/PlaceCard.tsx` — "Recommendation
  unavailable" state
- `web/components/trip-planner/RecommendationRing.tsx` — confidence band
  thresholds (80/60 instead of 65/40)

**Untracked (also pre-existing, from the Explore-relevance phase):**
- `api/ranking.py`
- `tests/test_ranking.py`
- `pipeline/modeling/rating_model.json`

Because `agent/crew.py` and `agent/tools.py` each carry both an earlier
phase's changes and this fix's changes in one uncommitted diff, they
cannot be cleanly split file-by-file — any commit boundary here has to be
a deliberate decision, not an automatic one. Nothing listed above is
committed.

## 8. Do not change

Everything documented above is a **working, verified baseline** as of
this snapshot. Future agents must not modify the recommendation model,
Explore ranking, Tier 1/Tier 2 discovery logic, duplicate-place
protection, or the Trip Planner fallback described here without first
**reproducing the specific issue** being fixed and **explaining its root
cause**. "This looks like it could be cleaner" is not sufficient
justification to touch code in this list — verified, working behavior
takes priority over stylistic preference.

## 9. Next phase

No implementation should start from this document alone. Recommended
order for whoever picks this up next:

A. Repository-wide diagnostic (confirm nothing else is silently broken)
B. UX / output quality review
C. Functional edge-case review
D. Performance / caching review
E. Final cleanup
F. Final testing
G. Commit / release
