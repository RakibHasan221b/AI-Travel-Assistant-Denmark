# Architecture

## Design principle: ML/rules before LLM calls

Every pipeline stage runs the cheapest available check first and only escalates
to an LLM when nothing cheaper can answer the question. Concretely:

1. A free keyword/location filter runs before any Reddit post reaches an LLM (Phase 4).
2. Once enough LLM-labeled examples exist, a cheap classifier is trained to
   replicate the LLM's decisions; the LLM is only called again when the
   classifier is unsure (Phase 5 — the ML→LLM→ML distillation loop).
3. GPT-4o (the most expensive model in the stack) is scoped to exactly one
   step — final RAG-grounded summaries (Phase 8) — everything else uses
   Groq or a local model.

## Backend

Streamlit through Phase 9 — a "fetch → compute → store → display" app doesn't
need a separate API, and it's the framework already used across prior
projects. A thin FastAPI service is added only in Phase 11, because the
CrewAI trip-planning crew needs stable session/process state that fights
Streamlit's rerun-on-every-interaction model. An optional Phase 14 adds a
Next.js/React/TypeScript frontend as an alternative client for that same API
(deployed on Vercel) — not required for the core pipeline, purely a bolt-on
once the API exists.

## Storage

Postgres only — no second database system. Semi-structured pipeline
artifacts (raw scraped JSON, raw LLM output) live in JSONB columns so
everything stays queryable with plain SQL instead of splitting storage
across systems. See `db/schema.sql` for the full serving-layer schema.

## Idempotent pipeline runs

Every ingestion run gets a run ID (`cph-YYYYMMDD-###`) recorded in
`pipeline_runs`, so scheduled jobs are safely resumable and backfillable
without duplicating data.

## Reddit is optional, not load-bearing

As of 2026, Reddit blocks anonymous `.json` access outright, and cloud/
datacenter IPs (including Vercel) are flagged even harder than a home IP —
so there is no free, reliable scraping path left, local or otherwise. Phase 4
now assumes the official Reddit OAuth API (free for non-commercial use,
~100 req/min, ~2-4 week manual approval) if Reddit is used at all, and it
runs as an optional, parallel-track phase: nothing in Phases 1-3 or 5-13
depends on it. The core pipeline is already complete without Reddit, using
OSM, Wikivoyage, opendata.dk, and the seed review set.

## Agent orchestration: CrewAI, not a single LangChain agent

The Phase 11 trip planner is built as a CrewAI crew (Place Scout, Concierge)
rather than one tool-calling loop. A third agent (Conditions Analyst) existed
early on to check weather/timing, but got folded into the Concierge as one
more tool call — its whole job was "call weather_conditions once, summarize
it," which doesn't need a dedicated reasoning agent, and every separate
agent CrewAI hands off to pays a real, measured fixed cost (its own
backstory + task description + a fresh opening LLM call) before doing any
real work — found live, ~1,400-1,600 tokens per run for a task that small.

Correction to the original plan: this doc originally said CrewAI's tools
would be LangChain-wrapped, on the assumption that CrewAI ran on LangChain
under the hood. Once Phase 11 was actually built against the installed
package (crewai 1.15.8), that turned out to be wrong — `Agent.tools` and
`Agent.llm` are typed to CrewAI's own `BaseTool`/`BaseLLM`, not LangChain's,
confirmed by inspecting the installed package rather than assumed. The four
tools (`agent/tools.py`) are built with `crewai.tools.tool`, and the LLM
(Groq) goes through CrewAI's own `LLM` class via litellm, not
`langchain-groq`. LangChain's one real, verified use in this project remains
Phase 8's RAG chain — Phase 11 doesn't use it at all.

Several real bugs were hit and fixed getting the Groq-backed crew working,
all documented inline in `agent/crew.py`: crewai 1.15.8 tags every LLM
message with an Anthropic-specific prompt-caching marker that's never
actually stripped for other providers (a genuine gap in the installed
package, confirmed by grepping its source — patched with a no-op monkeypatch
of `mark_cache_breakpoint`); the Place Scout could get stuck repeating the
exact same tool calls with identical arguments instead of recognizing it
already had its answer, found live burning ~12,000 tokens on a task that
needed one round of tool calls — fixed with an explicit stop-once-you-have-
results instruction; and `plan_trip()` no longer retries a rate-limit hit at
all, since a retry means re-running the whole crew from scratch, and on a
free tier this tight (12,000 TPM / 100,000 TPD) that's gambling a full run's
worth of tokens on a coin-flip rather than failing fast and letting a real
click try again.

## Recommendation scoring: from a static quality score to a live recommendation confidence

The original `quality_score` (Phase 9's XGBoost model, trained once on
structured OSM metadata alone, never looking at review text) correlated
weakly with real outcomes when actually checked against real ratings
(`r = 0.171`). Regression — predicting an exact 0–100 score from review
text — was tried and rejected on real evidence too (cross-validated R²
never beat a dummy baseline, at 188 *or* 527 labeled reviews; the target
itself is noisy, two people rate the same real experience differently, no
model can predict a number the text doesn't fully determine). Binary
classification — "would this be a good recommendation" — turned out to be
the task this data can actually support. See
`ai-denmark-explorer-real-time-scoring-full-report.md` (outside the repo,
Rakib's own copy) for the full investigation; the numbers below are the
final, shipped state.

**The current live flow, responsibilities kept deliberately separate:**

```
User query
    |
Retrieval — existing pgvector semantic search (Phase 6)
    |
Place candidates
    |
Recommendation scoring — agent/recommendation_service.py, live, every call:
  raw review text (reviews_raw, never a stored final score)
    -> DistilBERT sentiment signal (precomputed offline, see below)
       + MiniLM sentiment signal (computed live, every call — two
       separate scalars, not one — real evidence they catch different
       things: DistilBERT explicit opinion language, MiniLM implicit/
       contextual complaints a pure sentiment read misses)
    -> + structured features (category, price, opening hours,
       listing completeness, review count — NOT rating, that would
       leak the label into the input)
    -> pipeline/modeling/recommendation_classifier.json (XGBoost)
    -> recommendation_confidence (0-100) + recommendation_label
    |
Weather/context enrichment — Concierge's own weather_conditions call
    |
Concierge response — agent/crew.py's PlaceRecommendation/TripPlanOutput
```

**Real, measured result of the replacement** (173 real places with both an
old score and real review data): `old_quality_score` vs actual rating
`r=0.171`; `new recommendation_confidence` vs actual rating `r=0.668`. In
every one of the 10 largest real disagreement cases checked, the new
model matched the real outcome and the old one didn't.

**What this replaced, and what's deliberately unchanged:** `place_details`
(`agent/tools.py`) now computes `recommendation_confidence` live on every
call instead of reading `ml_predictions.quality_score` — the old value is
still logged server-side for ongoing comparison, never shown to a
traveler. `PlaceRecommendation.quality_score` (`agent/crew.py`) was kept
briefly as a deprecated `@computed_field` mirror while the frontend still
read the old name, then removed entirely once both the React and
Streamlit Trip Planner UIs were updated — see "Frontend: retiring
`quality_score` from the Trip Planner" below for that real cutover. The
Explore page and Stats Dashboard still read `ml_predictions.quality_score`
directly and are explicitly out of scope for this change, not an
oversight.

**DistilBERT relocated offline — real measurement forced this, not
preference.** A staged `psutil` memory test found DistilBERT's live
`transformers.pipeline()` peaking the process at ~804MB, 157% of
Render's 512MB free-tier limit — the same OOM class `agent/tools.py`
already documents once happening with `sentence-transformers`. Rather
than re-architecting the model, only *where* DistilBERT's inference runs
moved: `pipeline/modeling/precompute_distilbert_sentiment.py` now scores
every place's combined review text once, offline, and stores the result
in `ml_predictions` (`target='distilbert_sentiment'`, `confidence` field
doubles as the review count the score was computed from, so a place
getting new reviews later is detected as stale and rescored on the next
run — real, cheap reproducibility, not a one-off script). `place_details`
(`agent/tools.py`) now fetches that stored value with one more `LEFT
JOIN` and passes it into `predict_recommendation()`; `torch`/
`transformers` are no longer imported anywhere in the live request path.
MiniLM stays live via the shared `fastembed` instance — it was never the
problem.

**Verified, not assumed:** a direct live-vs-stored comparison across 25
real places found the stored score exactly reproduces a fresh live
DistilBERT call (0.000000 diff) once review concatenation order was made
deterministic (`ORDER BY review_id` — Postgres doesn't guarantee row
order without it, a latent nondeterminism that existed even before this
change). Re-running the full 173-place correlation check gave
`r=0.713` against real outcomes (previously 0.668; `old_quality_score`
still reproduces `r=0.171` exactly), confirming the relocation didn't
quietly change what the model actually predicts.

**Real, current memory measurement of the actual deployed entrypoint**
(`api.main`, staged with `psutil`, cold start + 30 sequential + 40
concurrent requests, values measured on Windows — Render's actual
containers run Linux): peaks around **~410MB cold, ~436-441MB under
concurrent load**, stable and reproducible across repeated runs, with
explicit singleton checks confirming no duplicate model instances.

**Production memory metrics could not be directly verified** — Render's
free tier does not expose application memory/CPU metrics at all (a paid
instance type is required to see that data), confirmed by checking the
live dashboard directly, not assumed. Stability is instead inferred from:
both `36eb938` and `7f3bf3d` deployed and started cleanly (clean
`Application startup complete` in ~1s in the real startup logs, no
import/artifact errors), no restarts or OOM kills observed, and the
repeated, reproducible local measurements above were taken against the
real production entrypoint, not an artificial import script. This is a
real, acknowledged gap, not a closed question — if it ever needs a hard
number, the honest path is a paid instance tier (to see Render's own
graph) or the app self-reporting via a real endpoint, not a guess.

## Live weather: from a periodic batch table to an on-demand Open-Meteo call

`agent/tools.py`'s `weather_conditions()` used to only read `weather_daily`,
a table populated purely by `ingestion/weather.py`'s periodic batch run
(historical backfill + a 7-day forecast snapshot). Real, current evidence
this had already gone stale in production: a live log capture showed
`"No weather data stored for 2026-08-10. Available range is 2025-01-01 to
2026-08-01"` — the batch job hadn't run in days, so even *today's* date had
no weather. The batch table is a real, useful cache; it just isn't a
substitute for asking Open-Meteo directly when the cache has gone cold.

**Fix: cache-aside, not a redesign.** On a `weather_daily` miss,
`_fetch_and_cache_live_weather()` calls Open-Meteo directly for that one
date — the archive endpoint for any past date (weather doesn't change, safe
to fetch on demand), the forecast endpoint for today through 16 days out
(Open-Meteo's own real free-tier limit, driven by what the API actually
returns for that date rather than a guessed cutoff) — then upserts the
result into `weather_daily` so the next request for the same date hits the
cache. `ingestion/weather.py`'s periodic batch run still exists and still
matters (keeps the common near-term window warm without a live call on
every single request), this just closes the gap when it's stale or a date
falls outside what it last covered.

**Kept honest, not guessed, past the real forecast horizon:** a date more
than 16 days out returns a plain "beyond Open-Meteo's real forecast
horizon, ask again closer to the date" message rather than inventing a
number. Real weather (temp/precip/wind) and the separate Outdoor Interest
Index (`visit_time_forecast`, Phase 10's trained model) are kept honestly
distinct — the Index is still only precomputed for whatever window
`forecast_interest.py`'s own batch run last covered, unrelated to weather's
new live-fetch range, and the tool's own text says so rather than implying
one covers the other.

Verified for real (pure Python, no LLM cost): today, +5 days, and a recent
past date all now return live-fetched, correctly-cached weather that was
previously missing; +25 days correctly returns the honest out-of-range
message. 50/50 tests passing (49 prior + one new offline test covering the
forecast-horizon cutoff, matching this suite's existing no-real-network/DB
convention).

## Live place-knowledge fallback: not "Wikipedia fallback" — ranked sources

Some curated places have zero linked `reviews_raw` text at all (never had
seed/Wikivoyage/review data ingested), so `place_details` had nothing to
score and honestly said so. `agent/tools.py`'s `place_details()` now
triggers a one-time live fallback when — and only when — a resolved place's
`review_texts` comes back empty: `_fetch_place_knowledge()` searches for
real, storable text, stores whatever's found as an ordinary `reviews_raw`
row (a new `reviews_raw.source_type` value, `'wikipedia'`, was added
alongside the existing ones — a real, small `ALTER TABLE` migration, not
just a docs change, applied to both `db/schema.sql` and the live database),
and returns it so the *same* request can score against it immediately, not
just the next one. Every later call sees the now-non-empty `review_texts`
from the normal `SELECT` and skips this branch entirely — the caching is
free, inherent in the existing query, not a separate mechanism.

**Deliberately not framed as "Wikipedia fallback."** The original plan led
with Wikipedia; real testing showed why that's the wrong default. A live
test against "Absalon" (a real landmark — a 1901 statue of the historical
bishop by Vilhelm Bissen) found a genuine, on-topic Wikipedia summary — but
it's a biography, not a description of the landmark or why a traveler
would visit, and the recommendation classifier scored it only 49%/"not
recommended" partly as a result. The system's actual job is answering "why
should I visit this place," which a real official site or a Copenhagen
tourism organization answers far better than an encyclopedia entry.

**The real hierarchy, in order:**
1. **The place's own official site** (from OSM's real `website` tag on
   that exact place — the strongest possible signal) — searched directly
   with a `site:`-scoped query, not just hoped for in general results.
2. **General Copenhagen search results**, ranked: known tourism-org domains
   (`visitcopenhagen.com`, `wonderfulcopenhagen.dk` — a short, hand-verified
   allowlist, not a heuristic guess) above generic snippets.
3. **Wikipedia**, only if nothing above found anything usable at all.

Every stored row's `raw_payload` records which real tier it came from
(`official_site` / `tourism_org` / `search_snippet` / the Wikipedia
extract itself) — so it's always possible to tell later exactly where a
description came from, not just that it came from "the web."

**A real bug caught and fixed by testing against a real, generically-named
place, not assumed safe:** "Abstrakt skulptur" (Danish for "abstract
sculpture" — not a unique proper noun) had its Serper results confidently
link three e-commerce listings for decorative sculpture *products*,
completely unrelated to the real public artwork — `web_enrichment.py`'s
existing `filter_results()` only screens snippet length and known
low-signal social domains, nothing about actual topical relevance. Fixed
with `_mentions_copenhagen_or_denmark()`, a real (if imperfect) relevance
gate requiring an explicit Copenhagen/Denmark/København mention in the
text itself, applied identically to Wikipedia and general-search results.
Re-tested after the fix: the tool now correctly finds nothing and says so
honestly, rather than storing wrong data confidently.

**A second, real limitation found and left honestly unresolved, not
papered over:** "Absalon" turned out to be genuinely ambiguous — the
historical bishop, a real *different* modern community venue (a
converted church), and a hotel all share the name. The database's actual
record is the 1901 statue (confirmed via its own OSM tags: sculptor
Vilhelm Bissen, a real official source `samlingen.koes.dk`), but that
niche Danish public-art-registry page turned out not to be indexed by
Google at all (`site:samlingen.koes.dk Absalon` returns zero results,
confirmed directly, not assumed) — so even the site-scoped search
couldn't find it, and the general search's tourism-org match was
confidently, plausibly, but *wrongly* about the unrelated community venue
instead. **This is the same underlying problem as the retrieval-ranking
limitation found during production verification** (multiple real,
distinct Copenhagen entities sharing similar/identical names) — not
fixed here, deliberately, to avoid scope creep into Phase B's territory;
flagged as a real, known gap rather than hidden. Verified the common case
works well instead: "AOC" (an unambiguous real restaurant with its own
well-indexed domain) correctly found three genuinely on-topic pages from
its own official site on the first, best-ranked tier.

52/52 tests passing (2 new offline tests covering the relevance gate and
the domain-tier ranking, matching this suite's existing no-real-network/DB
convention).

## Frontend: retiring `quality_score` from the Trip Planner

`web/lib/types.ts`'s `PlaceRecommendation` interface and
`web/components/trip-planner/PlaceCard.tsx` were the only two places in
the **React** frontend actually reading `quality_score` for the Trip
Planner path — confirmed by grepping the whole `web/` tree. Both now use
`recommendation_confidence`/`recommendation_label` directly; the card
shows "87% / RECOMMENDED" instead of "87/100 Quality", matching what the
model actually claims (a live recommendation estimate, not an objective
quality rating). `ExplorePlaceResult` and the Stats types keep their own,
separate `quality_score` fields untouched — the Explore page and Stats
Dashboard still correctly read `ml_predictions.quality_score` directly, by
design, unrelated to this rename.

With the React frontend no longer reading it, the deprecated
`quality_score` mirror was removed from both `agent/crew.py`'s
`PlaceRecommendation` (the `@computed_field` and its now-dead
`computed_field` import) and `api/main.py`'s independently-declared
response model — confirmed via a direct `model_dump()` check that
`quality_score` no longer appears in the serialized output at all.

**A real regression this created, caught by a later repo-wide sweep, not
by any test:** this project has *two* real, live-deployed Trip Planner
UIs (see the README's own two live links) — the React one, checked above,
and the original Streamlit one (`app/pages/2_Trip_Planner.py`), which was
never checked at the time. It read `place["quality_score"]` via direct
dict-key access, calling the same `/trip-plan` endpoint — removing the
mirror silently broke it (a real `KeyError` on any live request), and the
existing 52-test suite never caught it, because this page had zero test
coverage. Found and fixed in the same later sweep that caught the other
stale `quality_score` wording below — a genuine reminder that "the
frontend" can mean more than one deployed surface, and grepping one
directory isn't the same as checking every real consumer.

**Verified for real in a live browser, not just by reading the diff:**
ran both a local FastAPI server and the Next.js dev server together,
submitted the exact same request as an existing real cache entry ("tell
me about den lille havfrue") to confirm the null-confidence path renders
cleanly (no badge, no crash, no "undefined%"), then one new real request
("tell me about AOC restaurant") to confirm the populated path — the card
correctly showed "96% / RECOMMENDED", and the same response also carried
today's live-fetched weather ("high 22.9°C... 0.0mm precipitation"),
confirming both of today's backend changes work together end-to-end
through the actual UI. Zero console errors; `tsc --noEmit` clean.

**A real, separate mistake made and recovered during this phase, worth
recording honestly:** while wiring up local preview servers, a Bash check
for `.claude/launch.json` reported "no such file," which was wrongly
taken as proof the file didn't exist — it did, with two real working dev-
server configs (`web-trip-planner`, `static-guides`), and got overwritten.
`.claude/` is gitignored, so git history couldn't help. `web-trip-planner`
was fully recovered from an untouched second copy at `web/.claude/launch.json`;
`static-guides` was searched for across the entire project and filesystem
and never found — left out rather than guessed at. Lesson: a single
tool's "not found" is not proof of absence for a file inferred to matter,
especially gitignored local config with no git-history safety net.

## Production verification found a real, separate retrieval limitation — not a Phase A defect

Running one real end-to-end `/trip-plan` request against production
(after deploying `36eb938`/`7f3bf3d`) confirmed the recommendation
pipeline itself is deployed and working: `recommendation_confidence`/
`recommendation_label` are genuinely present in the real API response
(proving the `api/main.py` schema fix is live — previously these fields
would have been silently dropped, since Pydantic ignores undeclared
fields on construction). But for this specific request they came back
`null`, because the Place Scout never called `search_places`/
`top_quality_places` at all (confirmed directly in Render's logs — zero
real tool invocations, only the fallback `search_place_live` fired) and
fell back to the live Nominatim lookup, whose results are explicitly
never scored (`agent/crew.py`'s Concierge instructions require this).

**Root-caused with a direct, deterministic check (`search_places("Den
lille Havfrue")`), not another live agent run:** the real statue ranks
**#39** for a query of its own exact name — real, distinct Copenhagen
landmarks with similar names outrank it (`Den Genmodificerede Lille
Havfrue` at #1, `Den lille havfrue #2` at #11 — both confirmed via real
coordinates to be genuinely different places, not duplicate data, not a
data-quality bug). This is a **known, common limitation of embedding-only
retrieval**: semantic similarity optimizes for "what is this text about,"
not exact entity identity, so several real landmarks sharing similar
name/description text can outrank the one an exact-name query actually
means.

**This is a pre-existing Phase 6 (semantic search) limitation, unrelated
to and not introduced by the Phase A DistilBERT relocation** — Phase A's
own correctness is fully confirmed independent of this (the schema fix is
live, the null-handling path for live-lookup-only places works exactly as
designed). Tracked separately as Phase B below rather than folded into
Phase A's scope.

## Phase B (planned, not started): hybrid retrieval ranking

Add a lightweight lexical/fuzzy-match signal alongside the existing
pgvector semantic search, rather than trying to "fix" the embedding
itself — the standard pattern for named-entity queries in production
search systems (lexical + fuzzy + semantic + rerank, not a single
embedding score):

1. Normalize names (case-fold, strip accents/punctuation) for exact/
   fuzzy matching.
2. Boost heavily on an exact or near-exact normalized name match, e.g.
   `final_score = 0.7 * embedding_similarity + 0.3 * exact/fuzzy_name_score`.
3. Keep semantic similarity as the base signal for non-exact, descriptive
   queries — this isn't a replacement, only help for the exact-name case.
4. Re-evaluate against real canonical queries once built: "Den lille
   Havfrue", "Tivoli", "Nyhavn", "Rosenborg Castle" — confirm the intended
   place actually lands in the top few results, not just the exact-name
   case that surfaced this.
