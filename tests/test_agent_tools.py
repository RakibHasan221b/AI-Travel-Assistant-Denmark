"""Offline-safe: importing agent.tools must not touch the network or DB —
only calling a tool does. DATABASE_URL isn't read until a tool actually runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests

from agent.tools import (
    _coerce_limit,
    _domain_tier,
    _fetch_and_cache_live_weather,
    _MAX_FORECAST_DAYS,
    _mentions_copenhagen_or_denmark,
    haversine_km,
    place_details,
    search_place_live,
    search_places,
    search_places_near,
    set_trip_start,
    top_quality_places,
    travel_time_estimate,
    weather_conditions,
)

EXPECTED_TOOLS = [
    search_places, search_places_near, top_quality_places, place_details,
    travel_time_estimate, weather_conditions, search_place_live,
]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_tools_have_names_and_descriptions():
    for t in EXPECTED_TOOLS:
        assert t.name
        assert t.description


def test_weather_conditions_rejects_unparsable_date_without_hitting_the_network():
    result = weather_conditions.run(target_date="not-a-date")
    assert "Could not parse" in result


def test_live_weather_refuses_dates_beyond_open_meteos_real_forecast_horizon():
    # Real problem this guards: guessing weather for a date Open-Meteo
    # can't actually forecast yet, instead of saying so honestly. Checked
    # before any network/DB call — conn=None here would crash immediately
    # if that ordering ever regressed, which is the point of the test.
    from datetime import date, timedelta

    too_far = date.today() + timedelta(days=_MAX_FORECAST_DAYS + 5)
    result = _fetch_and_cache_live_weather(conn=None, d=too_far)
    assert result is None


def test_relevance_gate_requires_an_explicit_copenhagen_or_denmark_mention():
    # Real problem this guards, not hypothetical: a live test against the
    # generically-named landmark "Abstrakt skulptur" had Serper confidently
    # return three e-commerce listings for decorative sculpture products —
    # completely unrelated to the real place — because they only passed
    # length/domain filtering, nothing about actual topical relevance.
    assert _mentions_copenhagen_or_denmark("A statue in central Copenhagen, built in 1901.")
    assert _mentions_copenhagen_or_denmark("Beliggende i hjertet af København.")
    assert not _mentions_copenhagen_or_denmark("The Abstract Woman Sculpture adds elegance to any room.")


def test_domain_tier_ranks_official_site_above_tourism_org_above_generic():
    official = "restaurantaoc.dk"
    assert _domain_tier("https://restaurantaoc.dk/en/about", official) == 0
    assert _domain_tier("https://www.visitcopenhagen.com/copenhagen/x", official) == 1
    assert _domain_tier("https://some-random-blog.example/review", official) == 2
    # No official domain known for this place — tourism org still beats generic.
    assert _domain_tier("https://www.wonderfulcopenhagen.dk/x", None) == 1
    assert _domain_tier("https://some-random-blog.example/review", None) == 2


def test_travel_time_estimate_without_a_start_point_says_so_without_hitting_the_network():
    set_trip_start(None, None, "")
    result = travel_time_estimate.run(place_names="Den lille Havfrue")
    assert "No starting location was given" in result


def test_place_details_with_no_names_says_so_without_hitting_the_network():
    # Real root cause of a live rate-limit crash: place_details/
    # travel_time_estimate used to take one place per call, so the Concierge
    # called them once PER place — a request scouting several places burned
    # roughly that many extra round trips, each resending the growing
    # conversation, which is what pushed a single run over Groq's per-minute
    # token budget. Both tools now take a comma-separated batch instead.
    result = place_details.run(place_names="   ,  , ")
    assert result == "No place name given."


def test_travel_time_estimate_with_no_names_says_so_without_hitting_the_network():
    set_trip_start(55.68, 12.57, "Test Start")
    result = travel_time_estimate.run(place_names="")
    assert result == "No place name given."


def test_haversine_km_known_distance():
    # Copenhagen Central Station to the Little Mermaid statue, roughly 3.3 km
    # straight-line — a real, checkable distance, not an arbitrary number.
    dist = haversine_km(55.6726, 12.5646, 55.6929, 12.5993)
    assert 2.5 < dist < 4.0


def test_haversine_km_zero_distance_for_same_point():
    assert haversine_km(55.68, 12.57, 55.68, 12.57) == 0.0


def test_coerce_limit_accepts_a_real_int():
    assert _coerce_limit(5) == 5


def test_coerce_limit_accepts_a_stringified_int():
    # The actual live bug: Groq/Llama sometimes sends '"limit": "3"' (a JSON
    # string) instead of an integer, which Groq's own schema validation
    # used to reject outright before this coercion existed.
    assert _coerce_limit("3") == 3


def test_coerce_limit_falls_back_to_default_on_garbage():
    assert _coerce_limit("not-a-number") == 5
    assert _coerce_limit(None) == 5


def test_search_place_live_flags_a_found_result_as_not_in_the_curated_dataset(monkeypatch):
    # Real problem this guards: a live-lookup result must never be mistaken
    # for a curated, scored place downstream — the agent's honesty about
    # "no quality score" depends on this exact flag surviving in the string
    # it reads.
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse([{"name": "Reffen", "display_name": "Reffen, Copenhagen, Denmark"}]),
    )
    result = search_place_live.run(query="Reffen")
    assert "not in our curated dataset" in result
    assert "Reffen" in result


def test_search_place_live_reports_no_result_plainly(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse([]))
    result = search_place_live.run(query="somewhere that doesn't exist")
    assert "No live result found" in result


def test_search_place_live_handles_a_network_error_without_crashing(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", _raise)
    result = search_place_live.run(query="Christiania")
    assert "Live lookup failed" in result
