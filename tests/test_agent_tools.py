"""Offline-safe: importing agent.tools must not touch the network or DB —
only calling a tool does. DATABASE_URL isn't read until a tool actually runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests

from agent.tools import (
    _MAX_FORECAST_DAYS,
    _coerce_limit,
    _domain_tier,
    _fetch_and_cache_live_weather,
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
    from datetime import datetime, timedelta

    from agent.tools import _WEATHER_TZ

    too_far = datetime.now(_WEATHER_TZ).date() + timedelta(days=_MAX_FORECAST_DAYS + 5)
    row, error_kind = _fetch_and_cache_live_weather(conn=None, d=too_far)
    assert row is None
    assert error_kind == "out_of_range"


class _FakeHTTPErrorResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        error = requests.exceptions.HTTPError(f"{self.status_code} error")
        error.response = self
        raise error


def test_live_weather_distinguishes_a_real_429_from_other_provider_failures(monkeypatch):
    # Real problem this guards: a genuine Open-Meteo 429 was once reported
    # to a traveler identically to "beyond the forecast horizon" for a
    # date only 8 days out — a real, dishonest claim. rate_limited and
    # provider_unavailable must stay genuinely distinct error_kinds, not
    # just distinct in theory.
    import datetime as dt

    from agent.tools import _WEATHER_TZ

    today = dt.datetime.now(_WEATHER_TZ).date()

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeHTTPErrorResponse(429))
    row, error_kind = _fetch_and_cache_live_weather(conn=None, d=today)
    assert row is None
    assert error_kind == "rate_limited"

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeHTTPErrorResponse(500))
    row, error_kind = _fetch_and_cache_live_weather(conn=None, d=today)
    assert row is None
    assert error_kind == "provider_unavailable"

    def _raise_connection_error(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", _raise_connection_error)
    row, error_kind = _fetch_and_cache_live_weather(conn=None, d=today)
    assert row is None
    assert error_kind == "provider_unavailable"


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


def test_place_details_degrades_honestly_when_the_recommendation_model_fails(monkeypatch):
    # Real gap found while testing this case directly: place_details.run()
    # let a genuine model-load failure (e.g. a missing/corrupted model
    # file) propagate raw, uncaught — fine when CrewAI's own tool-call
    # wrapper happens to catch it during a real agent run, but
    # deterministic_trip_plan() (agent/crew.py, used when Groq itself is
    # down) calls _lookup_place_structured() directly with no such net,
    # so this would have surfaced as a raw 500 in exactly the "Groq is
    # down" moment the fallback exists for. Must degrade to real place
    # data with an honest, correctly-attributed reason — NOT "no review
    # text" when the place genuinely has reviews and the model is what
    # failed; that would be its own dishonest-error-message bug.
    import agent.recommendation_service as rec_service

    def _raise(*a, **k):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(rec_service, "predict_recommendation", _raise)

    result = place_details.run(place_names="Restaurant Klubben")
    assert "Restaurant Klubben" in result
    assert "Recommendation confidence: not available (the recommendation model is temporarily unavailable)" in result
    assert "AI summary:" in result  # everything else about the place still comes through


def test_place_details_caps_batch_size_regardless_of_how_many_names_are_passed(monkeypatch):
    # Real problem this guards: prompting the Scout to be selective about
    # candidate count was tried and found unreliable in real testing (it
    # still forwarded 13 loosely-related places for one query despite
    # explicit instructions). This is the deterministic backstop — each
    # extra place is a real predict_recommendation() call and, for
    # uncached places, a live web-lookup fetch, so an unbounded batch is
    # a genuine cost/reliability risk regardless of prompt compliance.
    # _lookup_place_structured is mocked out so this doesn't spend real
    # DB round-trips / live web-fallback fetches for 8 places just to
    # test the truncation count.
    import agent.tools as tools_module

    seen = []

    def _fake_lookup(cur, conn, place_name):
        seen.append(place_name)

    monkeypatch.setattr(tools_module, "_lookup_place_structured", _fake_lookup)

    names = ", ".join(f"Nonexistent Test Place {i}" for i in range(tools_module._MAX_PLACE_DETAILS_BATCH + 3))
    result = place_details.run(place_names=names)
    assert len(seen) == tools_module._MAX_PLACE_DETAILS_BATCH
    assert f"first {tools_module._MAX_PLACE_DETAILS_BATCH} places" in result


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
    # "no recommendation confidence" depends on this exact flag surviving
    # in the string it reads.
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
