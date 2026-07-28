"""Phase 11 — CrewAI trip-planning crew: three genuinely separate jobs
(finding places, checking timing/conditions, synthesizing a recommendation)
rather than one tool-calling loop. See docs/architecture.md.

LLM is Groq (llama-3.3-70b-versatile), not GPT-4o — GPT-4o stays scoped to
Phase 8's RAG summaries per this project's ML/rules-before-LLM-calls
principle. crewai.LLM talks to Groq via litellm's "groq/<model>" provider
prefix, reading GROQ_API_KEY from the environment.
"""

import os
import sys
import time

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

from agent.tools import place_details, search_places, top_quality_places, weather_conditions

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
        tools=[place_details],
        llm=llm,
        max_iter=MAX_AGENT_ITER,
        verbose=True,
    )

    scout_task = Task(
        description=(
            "Traveler request: {request}\n\n"
            "Find 3-5 candidate Copenhagen places matching this request. Use search_places "
            "for vibe/description matches and top_quality_places if the request is about "
            "finding the best-rated places. List only places your tools actually returned."
        ),
        expected_output="A short list of candidate places with category and neighborhood.",
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
            "call place_details on the 2-3 best-fitting candidates and write a final "
            "recommendation for the traveler's request: {request} (target date: {target_date}). "
            "Cite the quality score and, when available, the AI summary's sources for each "
            "place you recommend. Do not recommend a place you didn't call place_details on."
        ),
        expected_output=(
            "A final itinerary recommendation: 2-3 places with why each fits, their real "
            "quality score, and cited sources where available."
        ),
        agent=concierge,
        context=[scout_task, conditions_task],
    )

    return Crew(
        agents=[place_scout, conditions_analyst, concierge],
        tasks=[scout_task, conditions_task, concierge_task],
        process=Process.sequential,
        verbose=True,
    )


def plan_trip(request: str, target_date: str, _retry_wait_s: int = 15) -> str:
    """Retries once on a Groq TPM rate-limit hit — the free tier's ceiling
    sits close enough to this crew's per-run token usage that an occasional
    hit is expected, not exceptional (see GROQ_MODEL comment above)."""
    import litellm

    inputs = {"request": request, "target_date": target_date}
    try:
        return str(build_crew().kickoff(inputs=inputs))
    except litellm.RateLimitError:
        time.sleep(_retry_wait_s)
        return str(build_crew().kickoff(inputs=inputs))
