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
traveler. `PlaceRecommendation.quality_score` (`agent/crew.py`) is kept as
a deprecated `@computed_field` mirroring `recommendation_confidence`
exactly, purely so the not-yet-updated frontend doesn't break — remove it
once the frontend reads the new field directly. The Explore page and
Stats Dashboard still read `ml_predictions.quality_score` directly and
are explicitly out of scope for this change, not an oversight.

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
(`api.main`, staged with `psutil`, one real inference call, values
measured on Windows — Render's actual containers run Linux, so treat
this as a strong signal, not a Linux-exact number): peaks around
**~410MB**, stable across repeated calls (no growth after 10 calls).
Under Render's 512MB limit with real margin, though about 10MB over the
stricter <400MB safety target set for this phase — worth a real check
against Render's own memory graph after deploying, not assumed closed.
