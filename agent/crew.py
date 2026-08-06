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

LLM is Groq (llama-3.3-70b-versatile), not GPT-4o — GPT-4o stays scoped to
Phase 8's RAG summaries per this project's ML/rules-before-LLM-calls
principle. crewai.LLM talks to Groq via litellm's "groq/<model>" provider
prefix, reading GROQ_API_KEY from the environment.
"""

import logging
import os
import sys

import psycopg
import requests
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from agent.tools import (
    haversine_km,
    place_details,
    search_place_live,
    search_places,
    search_places_near,
    set_trip_start,
    top_quality_places,
    travel_fields,
    travel_time_estimate,
    weather_conditions,
)

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
# package actually calls that function (confirmed by grepping the source),
# so the marker reaches litellm and then Groq's strict OpenAI-compatible API
# verbatim, which rejects it as an unrecognized message property. Patching
# mark_cache_breakpoint() to a no-op is the minimal fix: it makes the marker
# behave exactly like the strip step that was supposed to run. Safe to
# remove once crewai actually wires up the strip step for non-Anthropic
# providers.
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda message: dict(message)

# llama-3.3-70b-versatile's free-tier TPM ceiling (12,000) sits right at the
# edge of what a full 3-agent run needs (~10,000-13,000 depending on how many
# candidates get scouted) — a clean run can succeed, but it's a tight margin,
# not comfortable headroom. Tried swapping to llama-3.1-8b-instant for more
# TPM room, but its function-calling is unreliable — verified live that it
# emits malformed tool-call syntax Groq's API rejects outright, so it's worse
# for this crew despite the extra headroom. Staying on 70b (proven correct
# tool-calling) and instead bounding max_iter per agent + retrying once on a
# rate-limit hit (see plan_trip) to keep the tight margin from being a hard
# failure.
GROQ_MODEL = "groq/llama-3.3-70b-versatile"
MAX_AGENT_ITER = 6

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


GROUNDING_RULE = (
    "Only state facts your tools actually returned. If a tool found nothing "
    "or a date is out of range, say so plainly instead of guessing or "
    "inventing detail — this matters more than sounding confident."
)


def build_llm():
    return {
        "model": GROQ_MODEL,
        "api_key": os.environ["GROQ_API_KEY"],
        "temperature": 0.3,
    }


def build_crew(llm_kwargs: dict | None = None) -> Crew:
    from crewai import LLM

    llm = LLM(**(llm_kwargs or build_llm()))

    place_scout = Agent(
        role="Place Scout",
        goal="Find Copenhagen places that genuinely match what the traveler is asking for.",
        backstory=(
            "You know Copenhagen's places inventory cold. You never suggest a place "
            "your tools didn't actually return. " + GROUNDING_RULE
        ),
        tools=[search_places, search_places_near, top_quality_places, search_place_live],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    concierge = Agent(
        role="Concierge",
        goal="Turn the scouted places and real conditions into one honest, specific recommendation.",
        backstory=(
            "You write the final answer a traveler actually reads. You pull rich detail "
            "(recommendation confidence, vibe cluster, rated aspects, AI summary with sources) for "
            "each place you recommend, check real weather and the Outdoor Interest Index "
            "before anyone commits to an itinerary, and you never oversell a place with "
            "thin evidence. " + GROUNDING_RULE
        ),
        tools=[place_details, travel_time_estimate, weather_conditions],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    scout_task = Task(
        description=(
            "Traveler request: {request}\n\n"
            "IMPORTANT: call each tool AT MOST ONCE per part of the request — never call the same "
            "tool with the same arguments a second time. The moment your tool results cover every "
            "part of the request, stop calling tools and give your final answer immediately; do not "
            "re-call a tool you already have a result from just to double-check it.\n\n"
            "First, work out what parts of the request you need to cover — a request can have more "
            "than one part (e.g. 'see the Little Mermaid AND have coffee nearby' has two parts: a "
            "named place, and an open-ended category). Cover every part, but only that many:\n"
            "- A named place (e.g. 'the Little Mermaid', 'Torvehallerne') → find just that place, "
            "confirm it exists. Do not add unrelated extra candidates for this part.\n"
            "- An open-ended part that says it's near/close to/around ANOTHER specific named place "
            "(e.g. 'coffee nearby' when a landmark was also named, 'a hotel near Torvehallerne') → "
            "use search_places_near with that other place as the anchor and a small limit (2-3, "
            "its default) unless the traveler clearly asked for more options — a few genuinely close "
            "picks beat a long list. This ranks by real geographic distance, not wording — do not "
            "use search_places for this, since text similarity alone doesn't mean something is "
            "actually close by.\n"
            "- An open-ended part with no reference point (a vibe, category, or 'best of' request "
            "with nothing to be near, e.g. 'a cozy cafe somewhere in the city') → find 1-3 real "
            "candidates with search_places (vibe/description match) or top_quality_places "
            "(best-rated).\n"
            "If the request is ONLY a named place with no open-ended part, return just that place. "
            "If it's ONLY open-ended with no named place, return 3-5 candidates for it. If it's both, "
            "return both parts — do not silently drop the open-ended part just because a named place "
            "was also mentioned. List only places your tools actually returned.\n\n"
            "If a NAMED place isn't found by search_places, search_places_near, or top_quality_places "
            "at all, try search_place_live as a last resort before giving up on it — it checks whether "
            "the place genuinely exists in Copenhagen even though it's outside our curated, scored "
            "dataset. Clearly note in your findings that it came from a live lookup, not the dataset."
        ),
        expected_output=(
            "Every distinct part of the request covered: the named place if one was given, and/or "
            "candidates for the open-ended part if one was given — each with category and "
            "neighborhood. Never fewer parts than the request actually asked for. If a named place "
            "was only found via search_place_live, say so explicitly instead of presenting it as an "
            "ordinary dataset result."
        ),
        agent=place_scout,
    )

    concierge_task = Task(
        description=(
            "IMPORTANT: never call the same tool with the same arguments twice — one batched "
            "place_details call, one weather_conditions call, and one batched travel_time_estimate "
            "call (if a start location was given) is enough; the moment you have those results, stop "
            "calling tools and write your final answer.\n\n"
            "Call weather_conditions once for target date {target_date} to get real weather and "
            "outdoor-interest conditions — if the date is out of the stored range, report that "
            "plainly in weather_summary instead of guessing.\n\n"
            "Using the Place Scout's candidates, call "
            "place_details ONCE with every place the Scout returned, comma-separated, for: "
            "{request} (target date: {target_date}) — never call place_details separately per "
            "place, that wastes tokens and risks a rate limit. If the Scout covered multiple parts "
            "of the request, include a result for each; don't collapse to one place. Fill name/"
            "category/neighborhood/opening_hours/recommendation_confidence/recommendation_label/"
            "vibe_cluster/summary/sources from place_details' actual output — null if unavailable, "
            "never guessed. recommendation_confidence and recommendation_label are exactly what "
            "place_details' \"Recommendation confidence\" line says, a live estimate of how likely "
            "this place is to be a good recommendation, not an objective quality rating — report "
            "them as returned, don't reinterpret or round them. Don't recommend a place you didn't "
            "get from place_details.\n\n"
            "EXCEPTION: for a place the Scout found only via search_place_live, leave it out of "
            "that place_details call (it isn't in the database) — use the live-lookup result as-is, "
            "leave recommendation_confidence/recommendation_label/vibe_cluster/summary/sources null.\n\n"
            "why_recommended and overall_note must sound like a real concierge talking to a "
            "traveler, not a system describing itself: never mention how a place was found (a "
            "tool, 'live lookup', 'our database'), and never comment on what data is/isn't "
            "available ('hours are known', 'not in our dataset'). Missing means simply not "
            "mentioning it, not announcing the gap.\n\n"
            "If the Scout noted a real search_places_near distance (e.g. '0.32 km from X'), set "
            "near_place and near_distance_km to that exact value — never invent or round it. Null "
            "for places found any other way.\n\n"
            "weather_summary: required, 1-2 sentences from your own weather_conditions call. "
            "overall_note: required, 2-4 sentences tying the recommendation together like a real "
            "concierge speaking — never empty. Weave the real weather from weather_conditions and "
            "the real distance/travel time from travel_time_estimate (if a starting point was "
            "given) naturally into this note as plain, warm sentences — 'it'll be a sunny "
            "22 degrees, an easy 15-minute walk from your hotel', not a separate report. If weather "
            "or distance genuinely couldn't be determined, simply don't mention it — never say "
            "'the weather forecast is not available' or similar; an omission reads as normal "
            "conversation, an announced gap reads as a system apologizing for itself.\n\n"
            "Traveler's starting point: {start_location}. If given (not 'not provided'), call "
            "travel_time_estimate ONCE with every place name comma-separated (same rule as "
            "place_details — one call, not one per place) and fill distance_km/walk_minutes/"
            "bike_minutes/travel_note. If not given, leave those four null — never guess a location."
        ),
        expected_output=(
            "A TripPlanOutput: every place the Scout found (not just one, unless the Scout only "
            "found one), each with its real recommendation confidence, hours, sources, and why it fits; "
            "near_place/near_distance_km filled in for places found via search_places_near; a "
            "weather summary for the target date; travel time fields filled in only if a "
            "starting point was given; and a real 2-4 sentence overall_note tying the "
            "recommendation together."
        ),
        agent=concierge,
        context=[scout_task],
        output_pydantic=TripPlanOutput,
    )

    return Crew(
        agents=[place_scout, concierge],
        tasks=[scout_task, concierge_task],
        process=Process.sequential,
        verbose=True,
    )


def _extract(crew_output) -> TripPlanOutput:
    """crew_output.pydantic is populated when output_pydantic parsing
    succeeds; falls back to wrapping the raw text in a single-field
    TripPlanOutput if the model's final answer couldn't be coerced into the
    schema — rare (instructor retries internally), but a fallback beats a
    hard crash on an otherwise-successful crew run."""
    if crew_output.pydantic is not None:
        return crew_output.pydantic
    log.warning("Concierge output didn't parse into TripPlanOutput, falling back to raw text")
    return TripPlanOutput(
        places=[],
        weather_summary="",
        overall_note=str(crew_output),
    )


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
    thresholds. near_place/near_distance_km are untouched — those measure
    distance to another recommended place, not to the traveler's start, so
    a different starting point doesn't change them."""
    with _cache_connect() as conn, conn.cursor() as cur:
        for place in result.get("places", []):
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


def plan_trip(request: str, target_date: str, start_location: str = "") -> dict:
    """Never retries a Groq rate-limit hit (TPM or TPD — Groq raises the
    identical litellm.RateLimitError for both). A retry here means re-running
    the ENTIRE 3-agent crew from scratch, not just the one failed call — so
    on a free tier this tight (12,000 TPM / 100,000 TPD), an automatic retry
    is gambling a full run's worth of tokens on a coin-flip, and losing that
    gamble can burn a quarter of the day's entire budget from a single user
    click. That trade only made sense while a real bug (see agent/tools.py's
    batching and the anti-loop task instructions below) was pushing every
    run right up against the ceiling; with that fixed, a clean run should
    fit comfortably, and a genuine rate-limit hit is now the exception, not
    the norm — worth failing fast and letting a human decide to click again,
    not worth spending another full run chasing it automatically.

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
    import litellm

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

    inputs = {
        "request": request,
        "target_date": target_date,
        "start_location": start_location.strip() or "not provided",
    }
    try:
        result = _extract(build_crew().kickoff(inputs=inputs)).model_dump()
    except litellm.RateLimitError as e:
        log.warning(f"Groq rate limit hit, not retrying (a retry re-runs the whole crew): {e}")
        raise
    except litellm.BadRequestError as e:
        # Occasional malformed tool-call generation (found live: Groq/Llama
        # sometimes emits <function=...></function> tags instead of proper
        # JSON, or gets a tool's argument types wrong, which Groq's strict
        # parser rejects as "tool_use_failed") — not a rate-limit issue, no
        # cooldown needed, just retry once immediately. Different failure
        # mode than the llama-3.1-8b-instant malformed-syntax issue noted
        # above (that one was consistent enough to rule the model out
        # entirely; this is an occasional glitch on the otherwise-reliable
        # 70b model, not guaranteed to be fixed by one retry).
        if "tool_use_failed" not in str(e):
            raise
        log.warning(f"Malformed tool-call generation, retrying once: {e}")
        result = _extract(build_crew().kickoff(inputs=inputs)).model_dump()

    _save_cache(request, target_date, start_location, result)

    return result
