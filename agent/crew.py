"""Phase 11 — CrewAI trip-planning crew: three genuinely separate jobs
(finding places, checking timing/conditions, synthesizing a recommendation)
rather than one tool-calling loop. See docs/architecture.md.

LLM is Groq (llama-3.3-70b-versatile), not GPT-4o — GPT-4o stays scoped to
Phase 8's RAG summaries per this project's ML/rules-before-LLM-calls
principle. crewai.LLM talks to Groq via litellm's "groq/<model>" provider
prefix, reading GROQ_API_KEY from the environment.
"""

import logging
import os
import sys
import time

import requests
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent.tools import (
    place_details,
    search_places,
    set_trip_start,
    top_quality_places,
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


class PlaceRecommendation(BaseModel):
    name: str
    category: str
    neighborhood: str = "unknown area"
    quality_score: float | None = Field(None, description="0-100, null if not available")
    vibe_cluster: str | None = None
    summary: str | None = Field(None, description="The AI-grounded summary, null if not available")
    sources: list[str] = Field(default_factory=list)
    distance_km: float | None = Field(None, description="null if no starting point was given")
    walk_minutes: int | None = None
    bike_minutes: int | None = None
    travel_note: str | None = Field(
        None, description="e.g. 'too far to walk comfortably, consider transit' — null if walking is fine or no starting point given"
    )
    why_recommended: str = Field(description="1-2 sentences on why this place fits the request")


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
        default="", description="Optional closing note — only for something that doesn't fit under a specific place, e.g. a caveat about data availability"
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
        tools=[search_places, top_quality_places],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    conditions_analyst = Agent(
        role="Conditions Analyst",
        goal="Assess weather and timing so the trip plan fits real conditions, not assumptions.",
        backstory=(
            "You check real weather and the Outdoor Interest Index before anyone commits "
            "to an itinerary. " + GROUNDING_RULE
        ),
        tools=[weather_conditions],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    concierge = Agent(
        role="Concierge",
        goal="Turn the scouted places and conditions into one honest, specific recommendation.",
        backstory=(
            "You write the final answer a traveler actually reads. You pull rich detail "
            "(quality score, vibe cluster, rated aspects, AI summary with sources) for "
            "each place you recommend, and you never oversell a place with thin evidence. "
            + GROUNDING_RULE
        ),
        tools=[place_details, travel_time_estimate],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    scout_task = Task(
        description=(
            "Traveler request: {request}\n\n"
            "If the traveler named one specific place (e.g. 'the Little Mermaid', 'Torvehallerne'), "
            "find just that place and confirm it exists — do not pad the list with unrelated extra "
            "candidates just because a broader search surfaces them. Only if the request is genuinely "
            "open-ended (a vibe, category, or 'best of' request, not one named place) should you find "
            "3-5 candidates: use search_places for vibe/description matches and top_quality_places if "
            "the request is about finding the best-rated places. List only places your tools actually "
            "returned."
        ),
        expected_output=(
            "Either one specific place (if the traveler named one) or a short list of 3-5 candidates "
            "(if the request was open-ended), each with category and neighborhood."
        ),
        agent=place_scout,
    )

    conditions_task = Task(
        description=(
            "Target date: {target_date}\n\n"
            "Check weather and outdoor-interest conditions for that date using "
            "weather_conditions. If the date is out of the stored range, report that "
            "plainly instead of guessing."
        ),
        expected_output="A short note on weather and whether conditions favor outdoor/indoor places.",
        agent=conditions_analyst,
    )

    concierge_task = Task(
        description=(
            "Using the Place Scout's candidates and the Conditions Analyst's timing note, "
            "call place_details on the 2-3 best-fitting candidates for the traveler's request: "
            "{request} (target date: {target_date}). Fill in each field from what place_details "
            "actually returned — leave a field null rather than guessing if it wasn't available. "
            "Do not recommend a place you didn't call place_details on.\n\n"
            "If the traveler named one specific place, your `places` list should contain only "
            "that place — do not pad it with other candidates the Scout found 'just in case.' "
            "Only include multiple places if the traveler's request was genuinely open-ended "
            "(a vibe or category, not one named place).\n\n"
            "weather_summary must always be filled in from the Conditions Analyst's note — "
            "never leave it empty.\n\n"
            "Traveler's starting point: {start_location}\n"
            "If a starting point was given (not 'not provided'), call travel_time_estimate on "
            "each place and fill in distance_km/walk_minutes/bike_minutes/travel_note from its "
            "result. If no starting point was given, leave those four fields null — do not "
            "guess a location."
        ),
        expected_output=(
            "A TripPlanOutput: each recommended place with its real quality score, sources, and "
            "why it fits, a weather summary for the target date, and travel time fields filled "
            "in only if a starting point was given."
        ),
        agent=concierge,
        context=[scout_task, conditions_task],
        output_pydantic=TripPlanOutput,
    )

    return Crew(
        agents=[place_scout, conditions_analyst, concierge],
        tasks=[scout_task, conditions_task, concierge_task],
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


def plan_trip(request: str, target_date: str, start_location: str = "", _retry_wait_s: int = 15) -> dict:
    """Retries once on a Groq TPM rate-limit hit — the free tier's ceiling
    sits close enough to this crew's per-run token usage that an occasional
    hit is expected, not exceptional (see GROQ_MODEL comment above).

    start_location is geocoded once here (zero LLM cost — plain HTTP call),
    not turned into its own agent tool call: the Concierge's
    travel_time_estimate tool reads the result via set_trip_start() instead
    of the agent having to geocode text itself, which would burn tokens on
    every single run instead of once per request.

    Returns a plain dict (TripPlanOutput.model_dump()), not the pydantic
    object itself — keeps the FastAPI layer decoupled from this module's
    internal schema class."""
    import litellm

    if start_location.strip():
        coords = geocode(start_location)
        if coords:
            set_trip_start(coords[0], coords[1], start_location)
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
        result = _extract(build_crew().kickoff(inputs=inputs))
    except litellm.RateLimitError:
        time.sleep(_retry_wait_s)
        result = _extract(build_crew().kickoff(inputs=inputs))
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
        result = _extract(build_crew().kickoff(inputs=inputs))

    return result.model_dump()
