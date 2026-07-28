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

The Phase 11 trip planner is naturally three distinct jobs — finding places,
checking timing/conditions, and synthesizing a recommendation — so it's built
as a CrewAI crew (Place Scout, Conditions Analyst, Concierge) rather than one
tool-calling loop.

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

Two real bugs were hit and worked around getting the Groq-backed crew
working, both documented inline in `agent/crew.py`: crewai 1.15.8 tags every
LLM message with an Anthropic-specific prompt-caching marker that's never
actually stripped for other providers (a genuine gap in the installed
package, confirmed by grepping its source — patched with a no-op monkeypatch
of `mark_cache_breakpoint`), and `llama-3.3-70b-versatile`'s free-tier Groq
rate limit (12,000 TPM) sits close enough to a full 3-agent run's token
usage that occasional rate-limit hits are expected, not exceptional — handled
with a bounded retry in `plan_trip()` rather than pretending it won't happen.
