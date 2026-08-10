"""Phase 11 — CrewAI trip-planning crew: two genuinely separate jobs
(finding places, synthesizing a recommendation) rather than one
tool-calling loop. See docs/architecture.md.

Weather/conditions used to be a third agent (Conditions Analyst) with its
own task. Folded into the Concierge as one more tool call instead — its
whole job was "call weather_conditions once, summarize it," which doesn't
need a dedicated reasoning agent, and every separate agent CrewAI hands
off to pays a real, measured fixed cost (its own backstory + task
description + a fresh opening LLM call) before it does any real work.
Found live: that fixed cost was ~1,400-1,600 tokens per run for a task
this small — cutting it directly reduces every single run's token bill,
not just repeated ones.

LLM is OpenAI (gpt-4o-mini by default, configurable via OPENAI_MODEL) — was
Groq (llama-3.3-70b-versatile) until Groq's free-tier 12,000 TPM ceiling
proved too unreliable in production (see plan_trip's own history of
rate-limit fixes). crewai routes "gpt-*" model names to its native OpenAI
provider (crewai.llms.providers.openai.completion.OpenAICompletion, using
the openai SDK directly, not litellm), reading OPENAI_API_KEY from the
environment. GPT-4o itself stays scoped to Phase 8's RAG summaries per
this project's ML/rules-before-LLM-calls principle — gpt-4o-mini is a
different, much cheaper model reserved for this crew.
"""

import logging
import os
import re
import sys
from contextvars import ContextVar

import openai
import psycopg
import requests
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from agent.intent import TripSpecification, execute_trip_specification
from agent.tools import (
    _WEATHER_ERROR_MESSAGES,
    _lookup_place_structured,
    _places_near,
    _resolve_place,
    _search_places_rows,
    _weather_structured,
    connect,
    get_cached_tool_calls,
    get_trip_start,
    haversine_km,
    reset_tool_call_cache,
    set_trip_start,
    travel_fields,
    weather_conditions,
)
from api.ranking import _CATEGORY_KEYWORDS, normalize_text

log = logging.getLogger("crew")

load_dotenv()

# Windows console default codepage can't encode the emoji crewai's verbose
# logging writes (crewai catches the resulting UnicodeEncodeError internally
# and just logs a warning, but it's noisy) — widen stdout/stderr to UTF-8.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# crewai 1.15.8 unconditionally tags every LLM message with a
# {"cache_breakpoint": true} marker meant for providers with explicit
# prompt-caching APIs (Anthropic). Its own crewai/llms/cache.py docstring
# says non-caching providers should have the marker stripped, and even
# defines strip_cache_breakpoint() to do it — but nothing in the installed
# package actually calls that function (confirmed by grepping the source,
# including crewai's native OpenAI provider — it doesn't strip this marker
# either), so it reaches OpenAI's strict API verbatim, which rejects it as
# an unrecognized message property, exactly as it did with Groq's. Patching
# mark_cache_breakpoint() to a no-op is the minimal fix: it makes the marker
# behave exactly like the strip step that was supposed to run. Safe to
# remove once crewai actually wires up the strip step for non-Anthropic
# providers.
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda message: dict(message)

# gpt-4o-mini is not a reasoning model (unlike the even-cheaper gpt-5-nano —
# tried first, but a real test call showed it can silently spend its whole
# max_tokens budget on invisible reasoning tokens and return nothing, which
# is exactly the unpredictable-token-usage failure this migration is meant
# to eliminate) and has a long, proven track record for reliable function/
# tool calling, which matters more here than shaving another fraction of a
# cent off an already-cheap request.
#
# Token/cost safeguards (small OpenAI credit balance — keep these
# conservative, not just "whatever the default happens to be"):
# - OPENAI_MAX_OUTPUT_TOKENS bounds every single LLM call's output, so one
#   call can never run away with a huge completion. 900 was too tight in
#   practice: verified live that the Concierge's structured JSON answer
#   (places, written first in TripPlanOutput's field order, then
#   weather_summary, then overall_note) hit exactly 900 completion_tokens
#   with a real, multi-place, start_location-given request — cutting off
#   mid-string in weather_summary ('"weather_summary": "On August 12,
#   2026, expect a high of 20.', no closing quote). instructor's own
#   retry-on-invalid-JSON then closed the syntax and completed
#   overall_note, but never went back to rewrite the now-truncated
#   weather_summary sentence, so the API response looked "successful"
#   while silently shipping half a sentence to the frontend. 1500 gives
#   real headroom above the 900 actually observed being hit.
# - MAX_LLM_CALLS_PER_REQUEST bounds total LLM calls across BOTH agents for
#   ONE trip-plan request (enforced in build_crew() below, not just
#   crewai's own per-agent max_iter) — hitting it raises
#   TripPlannerLLMUnavailable, which api/main.py treats exactly like an
#   OpenAI outage: fall back to the deterministic planner, never retry.
# - max_retries=0 in build_llm() disables the openai SDK's own automatic
#   retry-on-429/5xx behavior, so a single flaky call can't silently turn
#   into several billed attempts.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1500"))
MAX_LLM_CALLS_PER_REQUEST = int(os.environ.get("MAX_LLM_CALLS_PER_REQUEST", "6"))
MAX_AGENT_ITER = 3


class TripPlannerLLMUnavailable(Exception):
    """The single, narrow signal api/main.py catches to fall back to the
    deterministic planner — covers both a real OpenAI failure (rate limit,
    quota, timeout, connection, auth/config error) and this module's own
    MAX_LLM_CALLS_PER_REQUEST budget being hit. Deliberately not a bare
    isinstance() check against raw openai.* exception types at the API
    layer: crewai's own OpenAI provider re-wraps some of those (e.g.
    NotFoundError, APIConnectionError) into plain ValueError/ConnectionError
    internally, which would be unsafe to catch broadly several call frames
    away without risking hiding an unrelated real bug. Translating at the
    lowest point this module controls — the llm.call() wrapper in
    build_crew() — keeps that translation narrow and correct regardless of
    which concrete exception type any given OpenAI/crewai version raises."""


_llm_call_count: ContextVar[int] = ContextVar("_llm_call_count", default=0)

# Nominatim, same free/no-key service ingestion/osm_live_lookup.py already
# uses — single call per trip-plan request, well within its 1 req/s usage
# policy. Not a full geocoding pipeline: on any failure (typo, ambiguous
# text, network hiccup) this just leaves the trip without a start point
# rather than guessing or retrying, matching the honesty-over-confidence
# principle every tool here already follows.
def geocode(text: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{text}, Copenhagen, Denmark", "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "ai-denmark-explorer/0.1 (Copenhagen pilot, personal project)"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        log.warning(f"Geocoding failed for {text!r}: {e}")
        return None


# place_details() (agent/tools.py) uses filler words like "unknown" /
# "unclustered" / "not available (...)" so its text reads naturally for the
# LLM to consume — but see PlaceRecommendation's validator below for why
# these exact strings matter as a module-level constant, not a class attr.
_PLACEHOLDER_VALUES = {
    "unknown",
    "unclustered",
    "not available (no linked review text for this place).",
}


class PlaceRecommendation(BaseModel):
    name: str
    category: str
    neighborhood: str = "unknown area"
    opening_hours: str | None = Field(None, description="From place_details, null if unknown")
    # Renamed from quality_score after real evaluation showed why the name
    # was wrong, not just outdated: the old model (structured metadata only,
    # no review text) correlated weakly with real outcomes (r=0.171). The
    # model that replaced it (DistilBERT + MiniLM sentiment signals plus
    # structured features, computed live from a place's current review
    # text, never a stored number) correlated far more strongly (r=0.713) —
    # but it estimates "how likely is this place to be a good
    # recommendation," not an objective quality rating, so the field name
    # needed to say that honestly instead of implying a precision the model
    # doesn't have. The deprecated quality_score mirror that used to live
    # here has been removed now that the frontend reads this field directly.
    recommendation_confidence: float | None = Field(
        None, description="0-100, from place_details' live recommendation model — null if no review text was available"
    )
    recommendation_label: str | None = Field(
        None, description="'recommended' or 'not recommended', from place_details — null if recommendation_confidence is null"
    )
    vibe_cluster: str | None = None
    summary: str | None = Field(None, description="The AI-grounded summary, null if not available")
    sources: list[str] = Field(default_factory=list)
    distance_km: float | None = Field(
        None, description="Distance from the TRAVELER'S starting point (travel_time_estimate) — null if no starting point was given"
    )
    walk_minutes: int | None = None
    bike_minutes: int | None = None
    travel_note: str | None = Field(
        None, description="e.g. 'too far to walk comfortably, consider transit' — null if walking is fine or no starting point given"
    )
    near_place: str | None = Field(
        None,
        description="If this place was found via search_places_near (because the request said "
        "'near <some other place>'), the name of that other place — otherwise null.",
    )
    near_distance_km: float | None = Field(
        None,
        description="If near_place is set, the real distance to it from search_places_near's "
        "result — this is a DIFFERENT number from distance_km, which is distance from the "
        "traveler's own starting point, not from another recommended place. Null if near_place is null.",
    )
    why_recommended: str = Field(description="1-2 sentences on why this place fits the request")

    @field_validator("category", "neighborhood", mode="before")
    @classmethod
    def _coerce_null_category_and_neighborhood(cls, v, info):
        # Found live: Groq/Llama sometimes emits an explicit `null` for
        # category/neighborhood instead of the real value place_details
        # returned — likely for a place found only via search_place_live,
        # which doesn't carry a curated category. An explicit None crashes
        # the whole trip plan with a real ValidationError (str fields don't
        # accept it even with a default, since default only fills a
        # genuinely *missing* key). Same fix category as _coerce_null_sources
        # above: repair the one point the model's actual output is wrong,
        # rather than hoping the prompt prevents it every time.
        if v is not None:
            return v
        return "unknown area" if info.field_name == "neighborhood" else "place"

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_null_sources(cls, v):
        # Found live: Groq/Llama sometimes emits `"sources": null` for a
        # place with no cited sources, instead of `[]` — `default_factory`
        # only fills in a genuinely *missing* key, not an explicit null, so
        # this crashed the whole trip plan with a real ValidationError.
        # Same category of fix as _coerce_limit in agent/tools.py: coerce
        # at the one point the model's actual (if technically wrong) output
        # can still be repaired instead of hoping the prompt fixes it.
        return v if v is not None else []

    # Found live: place_details() (agent/tools.py) uses filler words like
    # "unknown"/"unclustered"/"not available (...)" so the TEXT it hands the
    # LLM reads naturally — but the Concierge sometimes copies that filler
    # word verbatim into the structured field instead of leaving it null,
    # so "Opening hours: unknown" became the literal string the traveler
    # saw, and "not available (no linked review text for this place)."
    # rendered as if it were a real summary. Since these UI fields are
    # already built to hide themselves entirely when null (no "Hours:
    # unknown" line at all), coercing the filler text back to null fixes
    # the display for free — no frontend change needed. (_PLACEHOLDER_VALUES
    # lives at module level, not as a class attribute — Pydantic intercepts
    # underscore-prefixed class attributes as private model fields.)
    @field_validator("opening_hours", "vibe_cluster", "summary", mode="before")
    @classmethod
    def _coerce_placeholder_text_to_null(cls, v):
        if isinstance(v, str) and v.strip().lower() in _PLACEHOLDER_VALUES:
            return None
        return v


class TripPlanOutput(BaseModel):
    """Structured trip-plan output — rendered as cards by the Streamlit
    frontend instead of a single free-text paragraph. Deliberately not
    asked for via prompt instructions alone ("please output JSON"): CrewAI's
    output_pydantic uses the instructor library to constrain/validate the
    LLM's actual response, which is far more reliable than hoping the model
    follows a formatting instruction on top of its narrative habits."""

    places: list[PlaceRecommendation]
    weather_summary: str = Field(description="1-2 sentences on conditions for the target date")
    overall_note: str = Field(
        default="",
        description="Required: 2-4 sentences tying the whole recommendation together in a warm, "
        "connected way, like a real concierge speaking — not just a leftover caveat.",
    )


class PlaceNarration(BaseModel):
    """One place's prose, and NOTHING else — the Concierge's output schema
    is deliberately this narrow (not the full PlaceRecommendation) so the
    LLM can never re-type a number agent/intent.py's execute_trip_
    specification() already computed deterministically. This is the
    structural fix for the exact risk the old single-shot TripPlanOutput
    schema carried: asking an LLM to reproduce recommendation_confidence/
    distance_km/near_distance_km itself in its structured output is asking
    it to transcribe data it didn't compute — precisely the failure mode
    _reconcile_near_relationships() used to exist to clean up after the
    fact. Here there is nothing numeric to transcribe in the first place."""

    name: str = Field(description="Must exactly match one of the place names given to you — never a place you weren't given")
    why_recommended: str = Field(description="1-2 sentences on why this place fits the request, grounded only in the facts given")


class ConciergeNarration(BaseModel):
    """The Concierge's entire output after the architecture change: prose
    only, grounded in already-computed, already-validated facts. weather_
    summary/overall_note are still free text, but every numeric/factual
    field a traveler sees (recommendation_confidence, distances,
    near_place, category, hours, ...) now comes from agent/intent.py's
    execute_trip_specification() and is merged in by plain Python
    (_execution_results_to_place_recommendations, below) — the LLM never
    gets a chance to alter it."""

    weather_summary: str = Field(description="1-2 sentences on real conditions for the target date, from your own weather_conditions call")
    overall_note: str = Field(
        default="",
        description="2-4 sentences tying the whole recommendation together, like a real concierge "
        "speaking — grounded only in the places and facts you were given, never inventing a place, "
        "score, distance, or relationship that wasn't already provided to you.",
    )
    place_narrations: list[PlaceNarration] = Field(default_factory=list)


GROUNDING_RULE = (
    "Only state facts your tools actually returned. If a tool found nothing "
    "or a date is out of range, say so plainly instead of guessing or "
    "inventing detail — this matters more than sounding confident."
)


def build_llm():
    return {
        "model": OPENAI_MODEL,
        "api_key": os.environ["OPENAI_API_KEY"],
        "temperature": 0.3,
        "max_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        # Disables the openai SDK's own automatic retry-on-429/5xx — see
        # the safeguards comment above MAX_AGENT_ITER for why.
        "max_retries": 0,
    }


_OPENAI_CALL_FAILURES = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APIStatusError,
    openai.APITimeoutError,
    openai.AuthenticationError,
    openai.NotFoundError,
    openai.InternalServerError,
    # crewai's native OpenAI provider re-wraps some of the above into these
    # plain builtins internally, before they ever reach this wrapper — see
    # TripPlannerLLMUnavailable's own docstring for why that re-wrapping
    # means this tuple has to include them too, not just the raw openai.*
    # types, to reliably reach the deterministic fallback.
    ConnectionError,
    ValueError,
)


def _instrument_llm(llm):
    """Wraps llm.call with the two hard safeguards this module owns: a
    per-request LLM-call budget, and translation of any OpenAI-layer
    failure into TripPlannerLLMUnavailable at the lowest point this code
    controls. Both agents in build_crew() share this one llm instance, so
    wrapping it once here covers the whole crew, not just one agent."""
    original_call = llm.call

    def _call(*args, **kwargs):
        count = _llm_call_count.get() + 1
        if count > MAX_LLM_CALLS_PER_REQUEST:
            raise TripPlannerLLMUnavailable(
                f"Trip Planner hit its {MAX_LLM_CALLS_PER_REQUEST}-call budget for this request"
            )
        _llm_call_count.set(count)
        try:
            return original_call(*args, **kwargs)
        except _OPENAI_CALL_FAILURES as e:
            raise TripPlannerLLMUnavailable(str(e)) from e

    llm.call = _call
    return llm


def build_intent_crew(llm) -> Crew:
    """The new intent-extraction crew — a single tool-less agent whose only
    job is to turn free-text into a validated TripSpecification
    (agent/intent.py). Deliberately no tools: the whole point of this
    architecture change is that the LLM no longer chooses which
    database/search function to call (see PROJECT_ARCHITECTURE_REPORT.md
    §16) — it only classifies intent into structured fields, which the
    backend then executes deterministically via
    agent.intent.execute_trip_specification(). Uses the exact same
    output_pydantic + instructor mechanism already proven on the old
    concierge_task, just applied one step earlier in the pipeline."""
    intent_analyst = Agent(
        role="Trip Intent Analyst",
        goal="Turn the traveler's free-text request into a precise, structured itinerary specification.",
        backstory=(
            "You understand Copenhagen trip requests deeply, but you never search a database or "
            "invent a distance yourself — your only job is to classify what the traveler actually "
            "said into structured fields the backend will execute exactly. You are especially careful "
            "never to let a sequence word like 'then' or 'afterwards' imply a spatial relationship — "
            "sequence and proximity are always separate questions."
        ),
        tools=[],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    intent_task = Task(
        description=(
            "Traveler request: {request}\n\n"
            "Break this request into one or more itinerary parts, in the order the traveler wants to "
            "visit them (sequence_index starting at 0). A request can have more than one part — e.g. "
            "'see the Little Mermaid AND have coffee nearby' has two parts: a named place, and an "
            "open-ended category.\n\n"
            "For EACH part, decide:\n"
            "- query: the traveler's own wording for this part.\n"
            "- named_place: true only if the traveler named one specific real place (e.g. 'the Little "
            "Mermaid', 'Torvehallerne', 'Rundetårn'); false for an open-ended category/vibe (e.g. 'a "
            "cozy cafe', 'somewhere to see art').\n"
            "- category: one of restaurant/cafe/hotel/landmark/bar if implied, else omit it — never "
            "invent one.\n"
            "- relation: decide this using ONLY explicit wording in the request, checked IN THIS "
            "ORDER — an explicit spatial word always wins over a bare sequence word, and a sequence "
            "word ALONE never implies a spatial relationship, no matter what:\n"
            "  1) EXPLICIT DISTANCE/FAR wording (far away, far from, somewhere distant, elsewhere, a "
            "different area, not near) → relation=far. This wins even if a sequence word or a named "
            "place also appears in the request. Example: 'see the Little Mermaid and THEN go to a "
            "cafe FAR AWAY' → the coffee part is relation=far, NOT near — the explicit 'far away' "
            "overrides the sequence word 'then' completely.\n"
            "  2) EXPLICIT PROXIMITY wording (near/nearby/close to/close by/around/next to/on the way "
            "to/within [a distance]/walking distance) → relation=near. If it points at another place "
            "the traveler actually named anywhere in the request, set anchor_query to that place's own "
            "wording (it only has to be named once, anywhere in the request). If it instead refers to "
            "the traveler's OWN starting point ('near me', 'close to where I'm staying'), set "
            "anchor_is_start_location=true instead, and leave anchor_query empty. If the traveler gave "
            "an approximate distance ('within 1 km' → 1.0, 'walking distance' → about 1.5), set "
            "max_distance_km to it. NEVER invent an anchor yourself — only use relation=near when the "
            "traveler's own words actually used a proximity word.\n"
            "  3) An explicit neighborhood/area name with no specific place named (e.g. 'around "
            "Nørrebro', 'somewhere in Vesterbro') → relation=area, with neighborhood set to that name.\n"
            "  4) NEITHER explicit distance/far wording NOR explicit proximity/area wording — "
            "including a part introduced ONLY by a sequence word with nothing else spatial (e.g. 'see "
            "the Little Mermaid and then have coffee' — 'then' alone states an order, not a location) "
            "→ relation=sequential for a non-first part with no reference point, or relation=primary "
            "for the main/first part of the request. Do not invent a near-relationship just because "
            "one wasn't explicitly ruled out.\n"
            "- min_distance_km: only if the traveler gave an explicit numeric lower bound for a "
            "relation=far part (e.g. 'at least 2 km away') — rare, omit otherwise.\n\n"
            "If the request is ONLY a named place with no open-ended part, return just that one part "
            "(relation=primary). If it's ONLY open-ended, return that one part. If it's both, return "
            "both parts — do not silently drop one just because the other was also mentioned."
        ),
        expected_output=(
            "A TripSpecification whose parts cover every distinct part of the request — the named "
            "place if one was given, and/or the open-ended part(s) if given — each with the correct "
            "relation chosen strictly from explicit wording, never inferred from a sequence word alone."
        ),
        agent=intent_analyst,
        output_pydantic=TripSpecification,
    )

    return Crew(agents=[intent_analyst], tasks=[intent_task], process=Process.sequential, verbose=True)


def build_concierge_crew(llm) -> Crew:
    """The narration-only Concierge — after the architecture change, every
    numeric/factual field (recommendation_confidence, distances,
    near_place, category, hours, ...) is already computed deterministically
    by agent.intent.execute_trip_specification() before this crew ever
    runs (see _run_structured_trip_plan below). The Concierge's only
    remaining job is: check real weather, then write grounded prose —
    output_pydantic=ConciergeNarration structurally prevents it from
    re-typing any number it was given, since ConciergeNarration has no
    numeric fields at all."""
    concierge = Agent(
        role="Concierge",
        goal="Write an honest, specific recommendation grounded only in the real places and facts you're given.",
        backstory=(
            "You write the final answer a traveler actually reads. Every place, score, distance, and "
            "relationship you're given is already real and already verified — your job is only to "
            "explain it warmly and check real weather, never to invent or restate a number "
            "differently than you were given it. " + GROUNDING_RULE
        ),
        tools=[weather_conditions],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    concierge_task = Task(
        description=(
            "Traveler request: {request}\n"
            "Traveler's starting point: {start_location}\n\n"
            "Call weather_conditions ONCE for target date {target_date} to get real weather and "
            "outdoor-interest conditions — if the date is out of the stored range, report that "
            "plainly in weather_summary instead of guessing. Do not call any other tool.\n\n"
            "Here are the real places already found and scored for this request — this is the "
            "complete, final set; do not add, remove, or rename any place, and do not restate or "
            "reinterpret any of their numbers:\n\n{scouted_places}\n\n"
            "For each place above, write a 1-2 sentence why_recommended that sounds like a real "
            "concierge talking to a traveler, not a system describing itself: never mention how a "
            "place was found (a tool, 'live lookup', 'our database'), and never comment on what data "
            "is/isn't available ('hours are known', 'not in our dataset') — missing means simply not "
            "mentioning it. If a place's distance is given as 'from X' (not 'from your start'), "
            "describe it relative to X in your prose — e.g. 'a 4-minute walk from the Little "
            "Mermaid' — never as distance from the traveler's own start; that would describe a "
            "different, less relevant question for that place.\n\n"
            "overall_note: required, 2-4 sentences tying the whole recommendation together like a "
            "real concierge speaking — weave in the real weather and any given distances naturally, "
            "but only facts you were actually given above or from your own weather_conditions call. "
            "Never invent a place, score, distance, or relationship that wasn't given to you."
        ),
        expected_output=(
            "A ConciergeNarration: a why_recommended for every place listed above (matching names "
            "exactly), a real weather_summary from your own weather_conditions call, and a genuine "
            "2-4 sentence overall_note."
        ),
        agent=concierge,
        output_pydantic=ConciergeNarration,
    )

    return Crew(agents=[concierge], tasks=[concierge_task], process=Process.sequential, verbose=True)


def _extract_spec(crew_output) -> TripSpecification:
    """Unlike _extract above, there's no reasonable fallback for a
    TripSpecification that failed to parse — a single-field "raw text"
    wrapper isn't executable by execute_trip_specification(). Raises
    TripPlannerLLMUnavailable instead (instructor already retries
    internally on a validation error; this only fires if every retry was
    exhausted), which api/main.py already treats as "fall back to the
    deterministic planner" — the same honest degradation path a real
    OpenAI outage takes."""
    if crew_output.pydantic is not None:
        return crew_output.pydantic
    raise TripPlannerLLMUnavailable("Intent extraction did not produce a valid TripSpecification")


def _extract_narration(crew_output) -> ConciergeNarration:
    """Same reasoning as _extract_spec — a ConciergeNarration that failed
    to parse has no safe partial-text fallback here (unlike the old
    single-shot TripPlanOutput, where the whole response was prose
    anyway); _run_structured_trip_plan's caller already has real,
    deterministically-computed places in hand and can synthesize an
    honest response from those alone (_synthesize_without_concierge)."""
    if crew_output.pydantic is not None:
        return crew_output.pydantic
    raise TripPlannerLLMUnavailable("Concierge narration did not produce valid output")


# Rolling window, not a hard daily reset — real Groq usage today (heavy
# live testing while building this crew) hit its own token cap, which is
# exactly the problem this cache exists to reduce for normal use.
CACHE_TTL_HOURS = 24


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _cache_connect():
    return psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=15)


def _get_exact_cache(request: str, target_date: str, start_location: str) -> dict | None:
    """Exact match on request+date+start_location within CACHE_TTL_HOURS —
    zero LLM cost on a hit. Real problem this fixes: re-submitting the same
    question (a user re-clicking, or repeat testing while building this
    feature) was spending real Groq tokens on an identical answer every
    single time."""
    with _cache_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT result_json FROM trip_plan_cache
            WHERE request_norm = %s AND target_date = %s AND start_location_norm = %s
              AND created_at > now() - interval '24 hours'
            ORDER BY created_at DESC LIMIT 1;
            """,
            (_normalize(request), target_date, _normalize(start_location)),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _get_same_request_cache(request: str, target_date: str) -> dict | None:
    """Same request+date, ANY start_location — used to adapt a cached plan
    for a new starting point without re-running the crew. The places,
    weather, and summary don't depend on where the traveler starts from;
    only the travel-time fields do, and those are recomputable with plain
    math (see _recompute_travel)."""
    with _cache_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT result_json FROM trip_plan_cache
            WHERE request_norm = %s AND target_date = %s
              AND created_at > now() - interval '24 hours'
            ORDER BY created_at DESC LIMIT 1;
            """,
            (_normalize(request), target_date),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _save_cache(request: str, target_date: str, start_location: str, result: dict) -> None:
    with _cache_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO trip_plan_cache (request_norm, target_date, start_location_norm, result_json)
            VALUES (%s, %s, %s, %s);
            """,
            (_normalize(request), target_date, _normalize(start_location), psycopg.types.json.Json(result)),
        )
        conn.commit()


def _recompute_travel(result: dict, start_lat: float, start_lon: float, start_label: str) -> dict:
    """Zero LLM cost: looks up each cached place's real coordinates and
    recalculates distance/walk/bike time from a NEW starting point, reusing
    travel_fields() so this stays consistent with the live tool's math and
    thresholds. Skips any place with near_place already set — its relevant
    distance is to that other recommended place, not to the traveler's own
    start, so a different starting point must never overwrite it with a
    start-relative distance_km (the same wrong-reference bug
    _reconcile_near_relationships() exists to prevent elsewhere)."""
    with _cache_connect() as conn, conn.cursor() as cur:
        for place in result.get("places", []):
            if place.get("near_place"):
                continue
            cur.execute(
                "SELECT lat, lon FROM places WHERE name ILIKE %(name)s "
                "ORDER BY (lower(name) = lower(%(exact)s)) DESC LIMIT 1;",
                {"name": f"%{place['name']}%", "exact": place["name"]},
            )
            row = cur.fetchone()
            if not row:
                continue
            dist_km = haversine_km(start_lat, start_lon, row[0], row[1])
            place.update(travel_fields(dist_km))
    log.info(f"Cache: reused places/weather for {start_label!r}, recomputed travel time with plain math")
    return result


def _place_recommendation_kwargs(
    detail: dict, why: str, start_coords: tuple[float, float] | None = None,
    lat: float | None = None, lon: float | None = None,
) -> dict:
    """Builds the common PlaceRecommendation kwargs from a
    _lookup_place_structured() result — shared by deterministic_trip_plan's
    whole-sentence path, its compound-request path (_compound_deterministic
    _places), and the cache-reuse path (_trip_plan_from_cached_results), so
    the three don't drift out of sync on which fields come from where.
    distance_km/walk_minutes/bike_minutes are the traveler's OWN
    start_location distance — never the "near X" distance, which callers
    set separately via near_place/near_distance_km."""
    kwargs = {
        "name": detail["name"],
        "category": detail["category"],
        "neighborhood": detail["neighborhood"] or "unknown area",
        "opening_hours": detail["opening_hours"],
        "recommendation_confidence": detail["recommendation_confidence"],
        "recommendation_label": detail["recommendation_label"],
        "vibe_cluster": detail["vibe_cluster"],
        "summary": detail["summary"],
        "sources": detail["sources"],
        "why_recommended": why,
    }
    if start_coords and lat is not None and lon is not None:
        dist_km = haversine_km(start_coords[0], start_coords[1], lat, lon)
        kwargs.update(travel_fields(dist_km))
    return kwargs


def _reconcile_near_relationships(result: dict) -> dict:
    """The real fix for a genuine class of bug: a place whose actual
    relationship is 'near the place visited before it' (e.g. a café near
    the Little Mermaid) was showing up with its distance FROM THE
    TRAVELER'S OWN START LOCATION instead — a real, correctly-calculated
    number, just the wrong reference point for that place. root cause:
    the Concierge asks travel_time_estimate for every place in one batched
    call, which unconditionally measures from the trip's start (see its
    own docstring) — and separately has to correctly transcribe the
    Scout's own near-distance text into near_place/near_distance_km,
    a fragile hand-off between two LLM calls this function doesn't rely
    on at all.

    Instead of trusting anything the LLM wrote, this looks at which
    search_places_near(anchor_place=...) calls the Scout ACTUALLY made
    this request (get_cached_tool_calls — the exact real kwargs, not a
    transcription) and independently re-derives the real anchor and real
    haversine distance for every one of its real candidates straight from
    the database — the same ground truth _compound_deterministic_places()
    already uses for the no-LLM fallback path. For any place in the final
    result that matches one of those real candidates, this overwrites
    near_place/near_distance_km with the recomputed real value and clears
    the start-relative distance_km/walk_minutes/bike_minutes/travel_note
    fields, since 'X km from your start' is a real but wrong-reference
    answer for a place whose relevant distance is to the place before it,
    not to where the trip began.

    General on purpose, not hardcoded to any place/category: works for
    whatever anchor and category the Scout actually searched near, for
    any request shape. A no-op when the Scout never called
    search_places_near at all (a plain single-destination request, or a
    request with no near-relationship) — existing single-destination
    behavior is untouched. Safe to call on every response path (the live
    crew run, the cache-reuse path, and — defensively — the deterministic
    fallback), since _tool_call_cache reflects this request's real tool
    calls regardless of which path ultimately produced the answer."""
    near_calls = get_cached_tool_calls("search_places_near")
    if not near_calls:
        return result

    anchor_cache: dict[str, tuple[str, float, float] | None] = {}
    near_lookup: dict[str, tuple[str, float]] = {}

    with connect() as conn, conn.cursor() as cur:
        for call in near_calls:
            anchor_place = str(call.get("anchor_place", "")).strip()
            if not anchor_place:
                continue
            category = call.get("category", "") or ""
            if anchor_place not in anchor_cache:
                resolved = _resolve_place(cur, anchor_place)
                anchor_cache[anchor_place] = (resolved[1], resolved[2], resolved[3]) if resolved else None
            anchor = anchor_cache[anchor_place]
            if anchor is None:
                continue
            anchor_name, anchor_lat, anchor_lon = anchor
            # A generous limit (not the Scout's own, possibly smaller, one)
            # so this matches any real nearby candidate the Concierge kept
            # in its final answer, even if it's not literally the closest
            # handful the Scout's own call happened to ask for.
            rows = _places_near(cur, anchor_lat, anchor_lon, category=category, exclude_name=anchor_name, limit=20)
            for r in rows:
                near_lookup[r["name"].lower()] = (anchor_name, round(r["distance_km"], 2))

    if not near_lookup:
        return result

    for place in result.get("places", []):
        match = _find_near_match(str(place.get("name", "")), near_lookup)
        if match is None:
            continue
        place["near_place"], place["near_distance_km"] = match
        place["distance_km"] = None
        place["walk_minutes"] = None
        place["bike_minutes"] = None
        place["travel_note"] = None

    return result


def _find_near_match(place_name: str, near_lookup: dict[str, tuple[str, float]]) -> tuple[str, float] | None:
    """Real gap found live: the Concierge's final answer sometimes shortens
    a place's real name (e.g. 'Terminalen kaffebar' for the database's real
    'Terminalen kaffebar - Seaside Toldboden') — an exact lookup silently
    misses that place, leaving its stale LLM-written distance_km
    uncorrected even though its near_place/near_distance_km happened to be
    right. Falls back to a substring match in either direction, which
    still only ever matches within near_lookup's own already-constrained,
    real, geographically-close candidate set — not a blanket fuzzy match
    against anything."""
    name_lower = place_name.lower().strip()
    if not name_lower:
        return None
    if name_lower in near_lookup:
        return near_lookup[name_lower]
    for candidate_name, value in near_lookup.items():
        if candidate_name in name_lower or name_lower in candidate_name:
            return value
    return None


def _ensure_start_distance(result: dict) -> dict:
    """Backfills distance_km/walk_minutes/bike_minutes/travel_note for any
    place that's missing them despite a real starting point being given —
    found live: the Concierge's own batched travel_time_estimate call
    sometimes only names the SECONDARY places (a reasonable-looking
    instinct once it correctly stopped treating them as start-relative —
    see _reconcile_near_relationships above — but it then also has to
    remember to separately ask for the PRIMARY place, which it doesn't
    always do). Rather than trying to make that one LLM call reliably
    cover every name every time, this deterministically fills the gap
    afterward using the exact same real, already-geocoded start
    coordinates (agent.tools.get_trip_start()) and the same haversine math
    travel_time_estimate itself uses — zero extra LLM cost, and it only
    ever fills a genuinely missing value, never overwrites one a tool
    call already set. Never touches a place with near_place set — that
    place's relevant distance is to another recommended place, not to the
    trip's start (see _reconcile_near_relationships)."""
    start = get_trip_start()
    if start["lat"] is None:
        return result

    missing = [
        p for p in result.get("places", [])
        if not p.get("near_place") and p.get("distance_km") is None
    ]
    if not missing:
        return result

    with connect() as conn, conn.cursor() as cur:
        for place in missing:
            resolved = _resolve_place(cur, str(place.get("name", "")))
            if resolved is None:
                continue
            _, _, lat, lon = resolved
            dist_km = haversine_km(start["lat"], start["lon"], lat, lon)
            place.update(travel_fields(dist_km))

    return result


def _weather_summary_text(weather: dict, target_date: str) -> str:
    """Formats _weather_structured()'s result into the same one/two-sentence
    summary shown across all three no-LLM response paths (deterministic
    whole-sentence, deterministic compound, and cache-reuse) — kept in one
    place so they can't drift into inconsistent wording."""
    if weather["ok"]:
        return (
            f"{weather['date']}: high {weather['temp_max_c']}°C / low {weather['temp_min_c']}°C, "
            f"{weather['precip_mm']}mm precipitation, {weather['wind_kph']}km/h wind."
        )
    if weather["error_kind"] == "unparsable_date":
        return f"Could not parse '{target_date}' as a date (expected YYYY-MM-DD)."
    return _WEATHER_ERROR_MESSAGES[weather["error_kind"]].format(
        date=weather.get("date", target_date), days_ahead=weather.get("days_ahead")
    )


# Splits "X and after Y" / "X and then Y" / "X then Y" into two halves.
# Deliberately tiny — covers the real compound phrasing this was built
# for ("wanna see little mermaid and after go to a nearby restaurant"),
# not an attempt at general clause parsing.
_COMPOUND_SPLIT_RE = re.compile(r"\band\s+(?:then\s+|after(?:wards)?\s+)?|\bthen\b", re.IGNORECASE)
_NEAR_RE = re.compile(r"\bnear(?:by)?\b", re.IGNORECASE)


def _split_compound_request(request: str) -> dict | None:
    """Best-effort split of a two-part request like 'wanna see little
    mermaid and after go to a nearby restaurant' into a PRIMARY named-place
    half and a SECONDARY category half, for deterministic_trip_plan()'s
    Groq-unavailable path. Deliberately tiny and literal: reuses api.
    ranking's existing _CATEGORY_KEYWORDS (the same map /explore and
    search_places already rely on) rather than adding any new taxonomy,
    parser, or LLM call. Requires the word "near"/"nearby" in the
    category-shaped half — that's the one relationship this handles for
    now (see agent/tools.py's _places_near for the actual distance logic).
    Returns None for anything that doesn't clearly split this way; callers
    fall back to treating the whole request as one semantic query, exactly
    like before this existed — the safe default for genuinely single-
    intent requests."""
    parts = _COMPOUND_SPLIT_RE.split(request, maxsplit=1)
    if len(parts) != 2:
        return None
    first, second = parts[0].strip(" ,."), parts[1].strip(" ,.")
    if not first or not second:
        return None

    def category_in(text: str) -> str | None:
        for tok in normalize_text(text).split():
            cat = _CATEGORY_KEYWORDS.get(tok)
            if cat:
                return cat
        return None

    cat_second, cat_first = category_in(second), category_in(first)
    if cat_second and not cat_first:
        primary_query, secondary_query, secondary_category = first, second, cat_second
    elif cat_first and not cat_second:
        primary_query, secondary_query, secondary_category = second, first, cat_first
    else:
        # Both halves (or neither) name a category — genuinely ambiguous,
        # don't guess which one is the "real" place.
        return None

    if not _NEAR_RE.search(secondary_query):
        return None

    return {
        "primary_query": primary_query,
        "secondary_query": secondary_query,
        "secondary_category": secondary_category,
        "relationship": "near",
    }


def _compound_deterministic_places(cur, conn, split: dict, start_coords: tuple[float, float] | None) -> list:
    """The 'near' half of deterministic_trip_plan()'s compound-request
    handling. Resolves the primary named place independently, via the
    same ranked search a plain single-intent request would use (never the
    whole compound sentence, which is what previously let the secondary
    category's words dilute/replace the primary place in the results).
    Ranks secondary candidates by REAL geographic distance FROM THE
    PRIMARY PLACE'S OWN COORDINATES (agent.tools._places_near — the same
    haversine logic search_places_near uses), never by semantic similarity
    and never using the traveler's own start_location as the anchor."""
    primary_candidates = _search_places_rows(split["primary_query"], limit=1)
    if not primary_candidates:
        return []
    primary = primary_candidates[0]
    primary_detail = _lookup_place_structured(cur, conn, primary["name"])
    if primary_detail is None:
        return []

    places = [
        PlaceRecommendation(
            **_place_recommendation_kwargs(
                primary_detail,
                why="The main place you asked to see.",
                start_coords=start_coords,
                lat=primary.get("lat"),
                lon=primary.get("lon"),
            )
        )
    ]

    if primary.get("lat") is not None and primary.get("lon") is not None:
        nearby_rows = _places_near(
            cur, primary["lat"], primary["lon"],
            category=split["secondary_category"],
            exclude_name=primary_detail["name"],
            limit=3,
        )
        for r in nearby_rows:
            detail = _lookup_place_structured(cur, conn, r["name"])
            if detail is None:
                continue
            # No start_coords here, deliberately: this place's relevant
            # distance is to the primary place (near_distance_km below),
            # not to the traveler's own start — passing start_coords would
            # compute a real but wrong-reference distance_km, the exact
            # bug _reconcile_near_relationships() exists to undo on the
            # live-LLM path (see its own docstring).
            kwargs = _place_recommendation_kwargs(
                detail,
                why=f"Near {primary_detail['name']}, matching your request for {split['secondary_category']}.",
                lat=r.get("lat"),
                lon=r.get("lon"),
            )
            kwargs["near_place"] = primary_detail["name"]
            kwargs["near_distance_km"] = round(r["distance_km"], 2)
            places.append(PlaceRecommendation(**kwargs))

    return places


def deterministic_trip_plan(request: str, target_date: str, start_location: str = "") -> dict:
    """The safety net api/main.py's /trip-plan calls when Groq is
    confirmed unavailable (a TPM rate limit, a malformed tool call that
    persisted through plan_trip's one retry, no capacity, or a transient
    outage) AND _trip_plan_from_cached_results() found nothing reusable —
    no LLM call at all, zero Groq tokens spent. Groq is the
    reasoning/orchestration layer here, not the ML recommendation engine:
    the real recommendation-confidence model (agent/recommendation_service
    .py, XGBoost + a numpy-only MiniLM signal) and the real place/weather
    data underneath were never Groq's job to begin with, so losing Groq
    doesn't have to mean losing them too.

    Runs the SAME real, deterministic lookups the Scout/Concierge
    normally orchestrate through an LLM — pgvector semantic search for
    candidates, then place_details' exact structured lookup (which calls
    predict_recommendation() directly) for each — just without the LLM
    doing the request-parsing, tool-calling, or narration. why_recommended
    /overall_note are honest, plain statements instead of LLM-authored
    prose, since there's no LLM here to write them, and overall_note says
    plainly that this is a degraded response rather than disguising it as
    a normal AI-narrated plan.

    Tries _split_compound_request() first (a real, if narrow, fix for a
    bug found live: a compound request like "wanna see little mermaid and
    after go to a nearby restaurant" used to be embedded as ONE whole-
    sentence query, and the restaurant words diluted the landmark clean out
    of the results). Falls back to the original whole-sentence semantic
    search — a reasonable, honest approximation for a degraded-mode
    response, not a substitute for a real crew run — whenever no compound
    structure is detected or the primary half doesn't resolve to a real
    place."""
    start_coords = geocode(start_location) if start_location.strip() else None

    split = _split_compound_request(request)
    places = []
    if split:
        with connect() as conn, conn.cursor() as cur:
            places = _compound_deterministic_places(cur, conn, split, start_coords)

    if not places:
        candidates = _search_places_rows(request, limit=5)
        if candidates:
            with connect() as conn, conn.cursor() as cur:
                for c in candidates:
                    detail = _lookup_place_structured(cur, conn, c["name"])
                    if detail is None:
                        continue
                    kwargs = _place_recommendation_kwargs(
                        detail,
                        why=(
                            "Matched from our database by relevance to your request — the AI "
                            "trip-planning assistant couldn't run a full personalized analysis "
                            "just now."
                        ),
                        start_coords=start_coords,
                        lat=c.get("lat"),
                        lon=c.get("lon"),
                    )
                    places.append(PlaceRecommendation(**kwargs))

    weather = _weather_structured(target_date)
    weather_summary = _weather_summary_text(weather, target_date)

    if places:
        overall_note = (
            "Our AI trip-planning assistant is temporarily unavailable (the language-model "
            "provider is at capacity), so these are real matching places from our database "
            "with their genuine recommendation confidence — not a personalized write-up. "
            "Try again shortly for the full AI-narrated experience."
        )
    else:
        overall_note = (
            "Our AI trip-planning assistant is temporarily unavailable, and no close database "
            "matches were found for this request either. Please try again shortly."
        )

    return TripPlanOutput(places=places, weather_summary=weather_summary, overall_note=overall_note).model_dump()


def _trip_plan_from_cached_results(request: str, target_date: str, start_location: str = "") -> dict | None:
    """Called by api/main.py's /trip-plan BEFORE falling back to
    deterministic_trip_plan(), when Groq fails after the crew's tool calls
    already succeeded. Real scenario found live: every tool call (search,
    place_details with real recommendation_confidence for all 4 places,
    weather) completed successfully, and ONLY the final LLM synthesis call
    hit Groq's daily token quota — deterministic_trip_plan() would have
    discarded all of that real, already-computed work and run a brand-new
    whole-sentence search instead, which is what let the primary named
    place drop out of a compound request's results.

    Reuses that work via agent.tools' per-request _tool_call_cache
    (reset at the start of every plan_trip() attempt, so whatever's in it
    here reflects only the CURRENT request — see reset_tool_call_cache()
    call sites in plan_trip() above): specifically, which place names the
    crew already successfully ran place_details for. Weather and
    travel-distance are recomputed via _weather_structured()/haversine_km
    rather than reading their own cache entries — both are cheap, local,
    non-LLM, non-Serper operations already backed by their own real
    caching (weather_daily table; haversine is pure math), so recomputing
    them here is not a second live provider call, just reusing the same
    functions deterministic_trip_plan() itself already relies on.

    Returns None (not a degraded TripPlanOutput) when the cache holds
    nothing useful, so the caller falls through to the normal
    deterministic path."""
    place_detail_calls = get_cached_tool_calls("place_details")
    if not place_detail_calls:
        return None

    seen_lower: set[str] = set()
    names: list[str] = []
    for kwargs in place_detail_calls:
        for n in str(kwargs.get("place_names", "")).split(","):
            n = n.strip()
            if n and n.lower() not in seen_lower:
                seen_lower.add(n.lower())
                names.append(n)
    if not names:
        return None

    start_coords = geocode(start_location) if start_location.strip() else None

    places = []
    with connect() as conn, conn.cursor() as cur:
        for name in names:
            detail = _lookup_place_structured(cur, conn, name)
            if detail is None:
                continue
            cur.execute(
                "SELECT lat, lon FROM places WHERE name ILIKE %(name)s "
                "ORDER BY (lower(name) = lower(%(exact)s)) DESC LIMIT 1;",
                {"name": f"%{name}%", "exact": name},
            )
            row = cur.fetchone()
            lat, lon = row if row else (None, None)
            kwargs = _place_recommendation_kwargs(
                detail,
                why=(
                    "Already found and scored earlier in this request — the AI trip-planning "
                    "assistant's final write-up step hit a temporary provider limit, so this "
                    "reuses the real results it had already gathered instead of starting over."
                ),
                start_coords=start_coords,
                lat=lat,
                lon=lon,
            )
            places.append(PlaceRecommendation(**kwargs))

    if not places:
        return None

    weather = _weather_structured(target_date)
    weather_summary = _weather_summary_text(weather, target_date)

    overall_note = (
        "Our AI trip-planning assistant hit a temporary limit with its language-model "
        "provider right at the final write-up step, after already gathering real results for "
        "your request — shown here as-is rather than being discarded for a fresh, less "
        "specific search."
    )

    result = TripPlanOutput(places=places, weather_summary=weather_summary, overall_note=overall_note).model_dump()
    # This loop gave every place a start-relative distance unconditionally
    # (it has no concept of "primary" vs "near X" place) — reconcile
    # against the Scout's real search_places_near calls from earlier in
    # this same request (still in _tool_call_cache) so a secondary place
    # gets its real near_place/near_distance_km instead of a wrong-
    # reference "X km from your start."
    return _ensure_start_distance(_reconcile_near_relationships(result))


def _format_execution_results_for_concierge(results: list[dict]) -> str:
    """Formats agent.intent.execute_trip_specification()'s real, already-
    scored results into text for concierge_task — the Concierge's
    replacement for calling place_details/travel_time_estimate itself.
    Mirrors place_details()'s own text-formatting style (agent/tools.py)
    so the Concierge sees the same shape of information it always has,
    just already-computed rather than fetched via a tool call."""
    if not results:
        return "No places were found matching this request."

    blocks = []
    for r in results:
        lines = [f"{r['name']} ({r['category']}, {r['neighborhood']})"]
        if r.get("opening_hours"):
            lines.append(f"Opening hours: {r['opening_hours']}")
        if r["recommendation_confidence"] is not None:
            lines.append(f"Recommendation confidence: {r['recommendation_confidence']}% ({r['recommendation_label']})")
        else:
            lines.append("Recommendation confidence: not available")
        if r.get("vibe_cluster"):
            lines.append(f"Vibe cluster: {r['vibe_cluster']}")
        if r.get("summary"):
            lines.append(f"AI summary: {r['summary']}")
        if r.get("near_place"):
            lines.append(
                f"Distance: {r['near_distance_km']:.2f} km from {r['near_place']} — this IS the "
                "relevant distance for this place; do not describe it as distance from the "
                "traveler's own start."
            )
        elif r.get("distance_km") is not None:
            note = f" — {r['travel_note']}" if r.get("travel_note") else ""
            lines.append(f"Distance from traveler's start: {r['distance_km']:.1f} km{note}")
        if r.get("execution_note"):
            lines.append(f"Note: {r['execution_note']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _match_narration(name: str, why_lookup: dict[str, str]) -> str | None:
    """Same exact/substring-both-ways matching strategy as _find_near_match
    above, applied to the Concierge's place_narrations instead of a
    near-distance lookup — handles the same real gap (the LLM shortening
    or lightly rewording a place's own name in its structured output)."""
    name_lower = name.lower().strip()
    if name_lower in why_lookup:
        return why_lookup[name_lower]
    for candidate, why in why_lookup.items():
        if candidate in name_lower or name_lower in candidate:
            return why
    return None


def _default_why_recommended(r: dict) -> str:
    """Plain, honest fallback prose — used only when no Concierge
    narration is available at all (_synthesize_without_concierge) or the
    Concierge's output didn't include this specific place by name."""
    if r.get("near_place"):
        return f"Found near {r['near_place']}, matching your request."
    return "Matches your request, with a real recommendation confidence from our database."


def _execution_results_to_place_recommendations(
    results: list[dict], narration: ConciergeNarration | None
) -> list[PlaceRecommendation]:
    """Merges agent.intent.execute_trip_specification()'s deterministic
    facts with the Concierge's prose (or an honest default if narration
    is None — the degraded no-Concierge path). This is the ONE place
    numeric fields and LLM-written prose come back together — every
    numeric field below comes from `results`, never from `narration`."""
    why_lookup = {pn.name.lower().strip(): pn.why_recommended for pn in narration.place_narrations} if narration else {}

    places = []
    for r in results:
        why = _match_narration(r["name"], why_lookup) or _default_why_recommended(r)
        places.append(
            PlaceRecommendation(
                name=r["name"],
                category=r["category"],
                neighborhood=r["neighborhood"],
                opening_hours=r.get("opening_hours"),
                recommendation_confidence=r["recommendation_confidence"],
                recommendation_label=r["recommendation_label"],
                vibe_cluster=r.get("vibe_cluster"),
                summary=r.get("summary"),
                sources=r.get("sources") or [],
                distance_km=r.get("distance_km"),
                walk_minutes=r.get("walk_minutes"),
                bike_minutes=r.get("bike_minutes"),
                travel_note=r.get("travel_note"),
                near_place=r.get("near_place"),
                near_distance_km=r.get("near_distance_km"),
                why_recommended=why,
            )
        )
    return places


# Real, live-observed gap: even with the structured intent layer making
# near_place/near_distance_km fully deterministic, the Concierge's own
# free-text prose sometimes still used casual proximity language for a
# place whose real, computed near_place was None — e.g. real overall_note
# text observed in testing: "...unwind at one of the cozy cafes nearby..."
# for cafes that were actually 3-6 km away (relation=sequential/far, not
# near). Fixed here deterministically, NOT by adding more prompt text
# (concierge_task's description is unchanged) — the whole point of this
# architecture is not to keep trusting the LLM to self-censor reliably.
_FALSE_PROXIMITY_PHRASES = (
    "nearby", "near by", "close by", "closeby", "short walk", "steps away",
    "stone's throw", "stones throw", "around the corner", "right next to",
    "next door", "walking distance", "a few minutes away", "just minutes away",
    # Found live during end-to-end verification (real overall_note/
    # why_recommended text for a relation=sequential place, e.g. "Its
    # proximity to the landmark...", "Located not far from the Little
    # Mermaid...") — same false-claim class, added to the same list
    # rather than a new mechanism.
    "proximity to", "not far from", "close to the",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Bare "near" is deliberately NOT in _FALSE_PROXIMITY_PHRASES above — found
# live: "conveniently located near several cozy cafes" for cafes actually
# 2.87-5.04 km away IS a false claim, but banning the word outright would
# also strip a real, unrelated geographic aside like "a prime location
# near the lakes" (real Hotel Nora prose, Test H — true and harmless).
# The two are told apart by what follows "near": a false claim always
# describes the itinerary itself (a generic category noun for the
# traveler's own recommended places, or one of those places by name); an
# unrelated landmark reference never does. This stays a local, deterministic
# text check — no NLP model, no extra LLM call.
_NEAR_WORD_RE = re.compile(r"\bnear\b(?!\s+(?:by|future))", re.IGNORECASE)
_ITINERARY_NOUN_WORDS = {
    "cafe", "cafes", "café", "cafés", "restaurant", "restaurants", "hotel", "hotels",
    "bar", "bars", "landmark", "landmarks", "spot", "spots", "place", "places",
    "option", "options", "stop", "stops",
}
_WORD_RE = re.compile(r"[a-zà-öø-ÿ']+")


def _bare_near_refers_to_itinerary(text: str, place_names: frozenset[str]) -> bool:
    """True only when a bare 'near' in the text is followed (within a
    short window) by a generic itinerary noun or one of the OTHER
    recommended places' own names — i.e. only when it's actually
    describing the itinerary relationship this guard polices, not an
    unrelated real-world landmark."""
    for m in _NEAR_WORD_RE.finditer(text):
        window = text[m.end(): m.end() + 40].lower()
        if set(_WORD_RE.findall(window)) & _ITINERARY_NOUN_WORDS:
            return True
        if any(name and name in window for name in place_names):
            return True
    return False


def _contains_proximity_claim(text: str, place_names: frozenset[str] = frozenset()) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in _FALSE_PROXIMITY_PHRASES):
        return True
    return _bare_near_refers_to_itinerary(text, place_names)


def _strip_proximity_sentences(text: str, place_names: frozenset[str] = frozenset()) -> str:
    """Removes only the sentence(s) making the false claim, keeping
    whatever other real content (weather, a true distance for a different
    place) the Concierge wrote — a smaller edit than discarding the whole
    field."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    kept = [s for s in sentences if not _contains_proximity_claim(s, place_names)]
    return " ".join(kept).strip()


def _enforce_narration_matches_deterministic_relationships(
    results: list[dict], narration: ConciergeNarration
) -> ConciergeNarration:
    """Deterministic backend guard: never edits a TRUE proximity claim (a
    place that genuinely has near_place set keeps its prose untouched) —
    only removes a claim attached to a place whose real, computed
    near_place is None. Same reasoning for overall_note: only scrubbed
    when NO place in the result set has a near relationship at all, since
    otherwise the phrase might correctly describe the one that does."""
    near_place_by_name = {r["name"].lower().strip(): bool(r.get("near_place")) for r in results}
    has_any_near_place = any(near_place_by_name.values())
    all_place_names = frozenset(near_place_by_name.keys())

    for pn in narration.place_narrations:
        if near_place_by_name.get(pn.name.lower().strip(), False):
            continue
        if _contains_proximity_claim(pn.why_recommended, all_place_names):
            match = next((r for r in results if r["name"].lower().strip() == pn.name.lower().strip()), None)
            pn.why_recommended = (
                _default_why_recommended(match) if match
                else _strip_proximity_sentences(pn.why_recommended, all_place_names)
            )

    if not has_any_near_place and _contains_proximity_claim(narration.overall_note, all_place_names):
        narration.overall_note = _strip_proximity_sentences(narration.overall_note, all_place_names) or (
            "Here's your plan based on real, matching places for your request."
        )

    return narration


def _synthesize_without_concierge(results: list[dict], target_date: str) -> dict:
    """Honest degraded response for the specific new failure mode this
    architecture introduces: intent extraction succeeded and
    execute_trip_specification() already produced real, scored places —
    but the Concierge's own final narration call then failed (OpenAI
    outage, or MAX_LLM_CALLS_PER_REQUEST hit on the 2nd of 2 calls this
    architecture now needs per request). Unlike the old
    _trip_plan_from_cached_results (which recovered partial work via
    agent.tools' tool-call cache — no longer populated the same way now
    that the Scout doesn't call search tools), this recovers directly
    from the real Python results already in hand, with zero LLM cost,
    same honest-about-being-degraded tone as deterministic_trip_plan()."""
    places = _execution_results_to_place_recommendations(results, narration=None)

    weather = _weather_structured(target_date)
    weather_summary = _weather_summary_text(weather, target_date)

    if places:
        overall_note = (
            "Our AI trip-planning assistant found and scored these real places for your request, but "
            "hit a temporary limit with its language-model provider right at the final write-up step "
            "— shown here with their real recommendation confidence and distances rather than being "
            "discarded. Try again shortly for the full AI-narrated experience."
        )
    else:
        overall_note = (
            "Our AI trip-planning assistant hit a temporary limit with its language-model provider, "
            "and no matching places were found for this request either. Please try again shortly."
        )

    return TripPlanOutput(places=places, weather_summary=weather_summary, overall_note=overall_note).model_dump()


def _run_structured_trip_plan(request: str, target_date: str, start_location: str) -> dict:
    """The new live-LLM path: intent extraction (1 LLM call, no tools) →
    deterministic backend execution (0 LLM calls — agent.intent.
    execute_trip_specification, real search/scoring/distance math) →
    Concierge narration (1 LLM call, weather_conditions only). Two LLM
    calls total per request, both structured, versus the old
    architecture's variable, tool-call-heavy Scout + Concierge exchange.

    Both crews share ONE instrumented llm instance so
    MAX_LLM_CALLS_PER_REQUEST/_llm_call_count (both request-scoped, see
    plan_trip below) correctly cover the whole request, not just one
    crew."""
    from crewai import LLM

    llm = _instrument_llm(LLM(**build_llm()))

    spec_output = build_intent_crew(llm).kickoff(inputs={"request": request})
    spec = _extract_spec(spec_output)
    log.info(f"TripSpecification: {spec.model_dump()}")

    execution_results = execute_trip_specification(spec)
    log.info(f"execute_trip_specification: {len(execution_results)} place(s) found/scored deterministically")

    concierge_inputs = {
        "request": request,
        "target_date": target_date,
        "start_location": start_location.strip() or "not provided",
        "scouted_places": _format_execution_results_for_concierge(execution_results),
    }
    try:
        narration_output = build_concierge_crew(llm).kickoff(inputs=concierge_inputs)
        narration = _extract_narration(narration_output)
        narration = _enforce_narration_matches_deterministic_relationships(execution_results, narration)
    except TripPlannerLLMUnavailable as e:
        log.warning(
            f"Intent extraction succeeded and {len(execution_results)} place(s) were already "
            f"deterministically found/scored, but the Concierge's final narration call failed ({e}) "
            "— synthesizing an honest response from the real results already in hand."
        )
        return _synthesize_without_concierge(execution_results, target_date)

    places = _execution_results_to_place_recommendations(execution_results, narration)
    return TripPlanOutput(
        places=places, weather_summary=narration.weather_summary, overall_note=narration.overall_note,
    ).model_dump()


def plan_trip(request: str, target_date: str, start_location: str = "") -> dict:
    """Never retries an OpenAI failure or a MAX_LLM_CALLS_PER_REQUEST hit —
    both surface as TripPlannerLLMUnavailable (see _run_structured_trip_plan/
    _instrument_llm).
    A retry here means re-running the ENTIRE crew from scratch, not just the
    one failed call, so on a small, fixed API budget an automatic retry is
    gambling a full run's worth of tokens rather than just failing fast and
    letting api/main.py's existing fallback answer with real database places
    immediately — cheaper and faster than a coin-flip second attempt.

    start_location is geocoded once here (zero LLM cost — plain HTTP call),
    not turned into its own agent tool call: the Concierge's
    travel_time_estimate tool reads the result via set_trip_start() instead
    of the agent having to geocode text itself, which would burn tokens on
    every single run instead of once per request.

    Checks the cache before spending any tokens: an exact repeat returns
    instantly; the same request+date with a different start_location
    reuses the cached places/weather and only recomputes travel time
    (pure math). Only a genuinely new request runs the real crew.

    Returns a plain dict (TripPlanOutput.model_dump()), not the pydantic
    object itself — keeps the FastAPI layer decoupled from this module's
    internal schema class."""
    exact = _get_exact_cache(request, target_date, start_location)
    if exact is not None:
        log.info("Cache: exact match, zero LLM cost")
        return exact

    if start_location.strip():
        coords = geocode(start_location)
        if coords:
            set_trip_start(coords[0], coords[1], start_location)
            same_request = _get_same_request_cache(request, target_date)
            if same_request is not None:
                result = _recompute_travel(same_request, coords[0], coords[1], start_location)
                _save_cache(request, target_date, start_location, result)
                return result
        else:
            log.warning(f"Could not geocode start_location={start_location!r}, proceeding without it")
            set_trip_start(None, None, start_location)
    else:
        set_trip_start(None, None, "")

    # Reset before every real crew execution — a fresh, request-scoped
    # slate for both the identical-tool-call cache (agent/tools.py) and the
    # MAX_LLM_CALLS_PER_REQUEST counter (_instrument_llm) — neither is a
    # cross-request cache/budget.
    reset_tool_call_cache()
    _llm_call_count.set(0)
    try:
        result = _run_structured_trip_plan(request, target_date, start_location)
    except TripPlannerLLMUnavailable as e:
        log.warning(f"OpenAI unavailable, not retrying (a retry re-runs the whole crew): {e}")
        raise

    result = _ensure_start_distance(_reconcile_near_relationships(result))
    _save_cache(request, target_date, start_location, result)

    return result
