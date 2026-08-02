"""Needs the isolated 'agent' venv (crewai) — see README's Phase 11 setup note."""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
