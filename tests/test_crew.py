"""Needs the isolated 'agent' venv (crewai) — see README's Phase 11 setup note."""

import sys
import time
from pathlib import Path

import litellm
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent.crew as crew_module
from agent.crew import PlaceRecommendation, TripPlanOutput, _normalize, geocode


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


def _rate_limit_error():
    return litellm.RateLimitError(message="rate limited", llm_provider="groq", model="llama-3.3-70b-versatile")


class _FakeCrewOutput:
    def __init__(self, pydantic_result):
        self.pydantic = pydantic_result


class _FakeCrew:
    def __init__(self, kickoff_fn):
        self._kickoff_fn = kickoff_fn

    def kickoff(self, inputs=None):
        return self._kickoff_fn()


def _fake_output():
    return _FakeCrewOutput(
        TripPlanOutput(places=[], weather_summary="sunny", overall_note="ok")
    )


def test_plan_trip_retries_and_succeeds_on_the_second_attempt(monkeypatch):
    # Real failure found live: a SINGLE retry (the old behavior) wasn't
    # always enough — this confirms a second retry can still rescue a run
    # that fails once after the first retry.
    monkeypatch.setattr(crew_module, "_get_exact_cache", lambda *a, **k: None)
    monkeypatch.setattr(crew_module, "_save_cache", lambda *a, **k: None)
    monkeypatch.setattr(crew_module.time, "sleep", lambda *a, **k: None)

    calls = {"n": 0}

    def kickoff_fn():
        calls["n"] += 1
        if calls["n"] < 3:  # fails on the initial attempt AND the first retry
            raise _rate_limit_error()
        return _fake_output()

    monkeypatch.setattr(crew_module, "build_crew", lambda: _FakeCrew(kickoff_fn))

    result = crew_module.plan_trip("test request", "2026-01-01")
    assert result["weather_summary"] == "sunny"
    assert calls["n"] == 3


def test_plan_trip_raises_after_exhausting_all_retries(monkeypatch):
    # Confirms retries are bounded, not infinite — a persistent rate limit
    # still surfaces as a real error instead of hanging or silently eating it.
    monkeypatch.setattr(crew_module, "_get_exact_cache", lambda *a, **k: None)
    monkeypatch.setattr(crew_module, "_save_cache", lambda *a, **k: None)
    monkeypatch.setattr(crew_module.time, "sleep", lambda *a, **k: None)

    calls = {"n": 0}

    def kickoff_fn():
        calls["n"] += 1
        raise _rate_limit_error()

    monkeypatch.setattr(crew_module, "build_crew", lambda: _FakeCrew(kickoff_fn))

    try:
        crew_module.plan_trip("test request", "2026-01-01")
        assert False, "expected RateLimitError to propagate"
    except litellm.RateLimitError:
        pass
    # 1 initial attempt + MAX_RATE_LIMIT_RETRIES retries
    assert calls["n"] == 1 + crew_module.MAX_RATE_LIMIT_RETRIES


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
