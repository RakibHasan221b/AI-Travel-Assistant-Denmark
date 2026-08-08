"""Needs the isolated 'agent' venv (crewai) — see README's Phase 11 setup note."""

import sys
from pathlib import Path

import litellm
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent.crew as crew_module
import agent.tools as tools_module
from agent.crew import PlaceRecommendation, TripPlanOutput, _normalize, geocode
from agent.tools import connect


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_geocode_returns_coords_on_a_real_looking_result(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse([{"lat": "55.6929", "lon": "12.5993"}]),
    )
    result = geocode("Den lille Havfrue")
    assert result == (55.6929, 12.5993)


def test_geocode_returns_none_on_no_results(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse([]))
    assert geocode("somewhere that doesn't exist") is None


def test_geocode_returns_none_on_network_error_instead_of_raising(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", _raise)
    assert geocode("Nyhavn") is None


def test_normalize_makes_cache_keys_insensitive_to_case_and_spacing():
    # Real problem this prevents: two requests that mean the same thing but
    # differ in casing/whitespace should still hit the same cache row.
    assert _normalize("  Vanlose  ") == _normalize("vanlose")
    assert _normalize("Coffee  Nearby") == _normalize("coffee nearby")


def test_normalize_empty_start_location_stays_a_stable_key():
    assert _normalize("") == _normalize("   ")


def test_place_recommendation_coerces_null_sources_to_empty_list():
    # Real failure found live: Groq/Llama sometimes emits `"sources": null`
    # for a place with no cited sources instead of `[]`. `list[str] =
    # Field(default_factory=list)` only fills in a genuinely *missing* key,
    # not an explicit null, so this crashed the whole trip plan with a real
    # pydantic ValidationError ("Input should be a valid list").
    place = PlaceRecommendation(
        name="Test Place", category="cafe", sources=None, why_recommended="test"
    )
    assert place.sources == []


def test_trip_plan_output_survives_multiple_places_with_null_sources():
    # The exact shape of the live failure: 6 places, all with null sources,
    # raised 6 stacked validation errors and failed the entire response.
    places = [
        {"name": f"Place {i}", "category": "cafe", "sources": None, "why_recommended": "test"}
        for i in range(6)
    ]
    output = TripPlanOutput(places=places, weather_summary="sunny")
    assert all(p.sources == [] for p in output.places)


def test_place_recommendation_coerces_placeholder_filler_text_to_null():
    # Real bug found live: place_details() (agent/tools.py) uses filler
    # words like "unknown"/"unclustered"/"not available (...)" so its TEXT
    # reads naturally for the LLM — but the Concierge copied that filler
    # text verbatim into the structured fields instead of leaving them
    # null, so travelers literally saw "Hours: unknown" and a summary that
    # just said "not available (no linked review text for this place)."
    place = PlaceRecommendation(
        name="Test",
        category="cafe",
        opening_hours="unknown",
        vibe_cluster="unclustered",
        summary="not available (no linked review text for this place).",
        why_recommended="test",
    )
    assert place.opening_hours is None
    assert place.vibe_cluster is None
    assert place.summary is None


def _rate_limit_error(message="rate limited"):
    return litellm.RateLimitError(message=message, llm_provider="groq", model="llama-3.3-70b-versatile")


class _FakeCrew:
    def __init__(self, kickoff_fn):
        self._kickoff_fn = kickoff_fn

    def kickoff(self, inputs=None):
        return self._kickoff_fn()


def test_plan_trip_never_retries_a_rate_limit_hit(monkeypatch):
    # A retry here means re-running the ENTIRE 3-agent crew, not just the
    # failed call — on a tight free-tier budget, an automatic retry gambles
    # a full run's worth of tokens on a coin-flip, and losing that gamble
    # can burn a large chunk of the day's quota from one user click.
    # Confirms build_crew is called exactly once, with no retry, regardless
    # of whether the hit was a per-minute (TPM) or per-day (TPD) limit —
    # Groq raises the identical litellm.RateLimitError for both.
    monkeypatch.setattr(crew_module, "_get_exact_cache", lambda *a, **k: None)
    monkeypatch.setattr(crew_module, "_save_cache", lambda *a, **k: None)

    calls = {"n": 0}

    def kickoff_fn():
        calls["n"] += 1
        raise _rate_limit_error(
            "Rate limit reached ... on tokens per day (TPD): Limit 100000, "
            "Used 98990, Requested 2882. Please try again in 26m57.408s."
        )

    monkeypatch.setattr(crew_module, "build_crew", lambda: _FakeCrew(kickoff_fn))

    try:
        crew_module.plan_trip("test request", "2026-01-01")
        assert False, "expected RateLimitError to propagate"
    except litellm.RateLimitError:
        pass
    assert calls["n"] == 1


def test_place_recommendation_keeps_real_values_matching_placeholder_case_insensitively():
    # Guards against the coercion being too aggressive/case-sensitive in
    # either direction.
    place = PlaceRecommendation(
        name="Test", category="cafe", opening_hours="UNKNOWN", why_recommended="test"
    )
    assert place.opening_hours is None

    place2 = PlaceRecommendation(
        name="Test2", category="cafe", opening_hours="Mo-Fr 08:00-18:00", why_recommended="test"
    )
    assert place2.opening_hours == "Mo-Fr 08:00-18:00"


def test_cached_place_details_are_reused_instead_of_a_fresh_whole_sentence_search(monkeypatch):
    # Real bug found live: Groq's daily token quota was exhausted right at
    # the final synthesis call, AFTER every tool call (including
    # place_details, with a real recommendation_confidence) had already
    # succeeded — deterministic_trip_plan() threw all of that real work
    # away and ran a brand-new whole-sentence semantic search instead,
    # which let "Den lille Havfrue" drop out of a compound request's
    # results entirely. _trip_plan_from_cached_results() must recover the
    # already-computed place instead of triggering any new search.
    tools_module.reset_tool_call_cache()
    tools_module._tool_call_cache[
        ("place_details", (("place_names", "Den lille Havfrue"),))
    ] = "irrelevant cached text — only the kwargs are read"

    def _fail_if_called(*a, **k):
        raise AssertionError("must not run a fresh semantic search when cached results exist")

    monkeypatch.setattr(crew_module, "_search_places_rows", _fail_if_called)

    result = crew_module._trip_plan_from_cached_results(
        "wanna see little mermaid and after go to a nearby restaurant", "2026-01-01", ""
    )

    assert result is not None
    assert [p["name"] for p in result["places"]] == ["Den lille Havfrue"]
    assert result["places"][0]["recommendation_confidence"] is not None


def test_split_compound_request_keeps_little_mermaid_primary_and_restaurant_secondary():
    # The exact real phrase the user hit live: a compound "see X and then Y
    # nearby" request must split into a named-place primary half and a
    # category-shaped secondary half, not be treated as one semantic query.
    split = crew_module._split_compound_request(
        "wanna see little mermaid and after go to a nearby restaurant"
    )
    assert split is not None
    assert split["primary_query"] == "wanna see little mermaid"
    assert split["secondary_query"] == "go to a nearby restaurant"
    assert split["secondary_category"] == "restaurant"
    assert split["relationship"] == "near"


def test_split_compound_request_returns_none_for_a_plain_single_intent_request():
    # Safe default: a request with no compound structure (or no "near"
    # relationship) must fall through to the whole-sentence search, not
    # get force-split on a guess.
    assert crew_module._split_compound_request("cozy quiet cafe good for working") is None


def test_compound_deterministic_places_ranks_secondary_by_real_distance_from_primary():
    # Requirement: "nearby" must be resolved from REAL coordinates of the
    # PRIMARY place (never the traveler's own start_location, never
    # semantic similarity) — verified here against the real database.
    split = crew_module._split_compound_request(
        "wanna see little mermaid and after go to a nearby restaurant"
    )
    assert split is not None

    with connect() as conn, conn.cursor() as cur:
        places = crew_module._compound_deterministic_places(cur, conn, split, start_coords=None)

    assert places[0].name == "Den lille Havfrue"
    secondary = places[1:]
    assert len(secondary) >= 2, "expected real curated restaurants near the Little Mermaid"
    assert all(p.near_place == "Den lille Havfrue" for p in secondary)

    distances = [p.near_distance_km for p in secondary]
    assert distances == sorted(distances), "secondary places must be ranked nearest-first"

    # Cross-check each reported distance against real haversine math from
    # the primary place's own coordinates.
    primary_lat, primary_lon = 55.6928661, 12.5992896
    with connect() as conn, conn.cursor() as cur:
        for p in secondary:
            cur.execute("SELECT lat, lon FROM places WHERE name = %s;", (p.name,))
            lat, lon = cur.fetchone()
            expected = tools_module.haversine_km(primary_lat, primary_lon, lat, lon)
            assert abs(p.near_distance_km - round(expected, 2)) < 0.01
