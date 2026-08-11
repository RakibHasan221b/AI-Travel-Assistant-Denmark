"""Offline-safe: importing agent.tools must not touch the network or DB —
only calling a tool does. DATABASE_URL isn't read until a tool actually runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests

from agent.tools import (
    _MAX_FORECAST_DAYS,
    _coerce_limit,
    _discover_live_place,
    _domain_tier,
    _fetch_and_cache_live_weather,
    _find_curated_match,
    _lookup_place_structured,
    _map_nominatim_category,
    _mentions_copenhagen_or_denmark,
    connect,
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
    # in the string it reads. No category/type in this mock (unlike a real
    # Nominatim response) — deliberately exercises the "category unclear,
    # can't safely persist" branch, not full live discovery, keeping this
    # test's scope narrow and DB-free.
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse([{
            "name": "Reffen", "display_name": "Reffen, Copenhagen, Denmark",
            "lat": "55.6989", "lon": "12.6122",
        }]),
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


def test_map_nominatim_category_maps_known_amenity_and_tourism_pairs():
    assert _map_nominatim_category("amenity", "cafe") == "cafe"
    assert _map_nominatim_category("amenity", "restaurant") == "restaurant"
    assert _map_nominatim_category("amenity", "bar") == "bar"
    assert _map_nominatim_category("tourism", "hotel") == "hotel"
    assert _map_nominatim_category("tourism", "artwork") == "landmark"


def test_map_nominatim_category_falls_back_by_class_for_historic_and_leisure():
    assert _map_nominatim_category("historic", "monument") == "landmark"
    assert _map_nominatim_category("leisure", "park") == "landmark"


def test_map_nominatim_category_returns_none_for_an_unmapped_pair():
    # Real requirement this guards: never guess a default category just to
    # make a row insertable — category is a real model feature, and an
    # unmapped OSM type (e.g. a shop) has no confident answer.
    assert _map_nominatim_category("shop", "bakery") is None


def test_find_curated_match_finds_the_real_little_mermaid_by_exact_osm_id():
    # Real, live-verified case: Nominatim's own real osm_id for "The
    # Little Mermaid" (node/25074274) is the exact same osm_id the
    # curated database already stores for "Den lille Havfrue" — a real
    # entity match a coordinate/name heuristic wouldn't need to guess at.
    with connect() as conn, conn.cursor() as cur:
        match = _find_curated_match(cur, "node", 25074274, 55.6928661, 12.5992896)
    assert match is not None
    assert match[1] == "Den lille Havfrue"


def test_find_curated_match_returns_none_far_from_any_curated_place():
    with connect() as conn, conn.cursor() as cur:
        match = _find_curated_match(cur, None, None, 0.0, 0.0)  # middle of the ocean
    assert match is None


def _cleanup_discovered_place(place_id) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT review_id FROM reviews_raw WHERE place_id = %s;", (place_id,))
        for (review_id,) in cur.fetchall():
            cur.execute("DELETE FROM place_mentions WHERE review_id = %s;", (review_id,))
        cur.execute("DELETE FROM reviews_raw WHERE place_id = %s;", (place_id,))
        cur.execute("DELETE FROM places WHERE place_id = %s;", (place_id,))
        conn.commit()


def test_discover_live_place_persists_and_scores_when_real_evidence_is_found(monkeypatch):
    # Core requirement B: a new place with fresh review data receives a
    # real ML score computed through the exact same pipeline as any
    # curated place — not an LLM-generated or invented number.
    from ingestion import web_enrichment

    monkeypatch.setattr(
        web_enrichment, "search_web",
        lambda q: [{
            "title": "Test Discovery Place - Copenhagen guide",
            "link": "https://example.com/test-discovery-place",
            "snippet": "A wonderful test venue in Copenhagen with great reviews and a cozy, welcoming atmosphere for everyone.",
        }],
    )

    place_id = None
    try:
        with connect() as conn:
            result = _discover_live_place(
                conn, "Test Discovery Place XYZ", "restaurant", 55.68, 12.57, None, None
            )
        assert result is not None
        place_id, found_texts = result
        assert len(found_texts) >= 1

        with connect() as conn, conn.cursor() as cur:
            detail = _lookup_place_structured(cur, conn, "Test Discovery Place XYZ")
        assert detail is not None
        assert detail["recommendation_confidence"] is not None
        assert detail["recommendation_label"] in ("recommended", "not recommended")

        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT data_status, source_url FROM places WHERE place_id = %s;", (place_id,))
            data_status, source_url = cur.fetchone()
        assert data_status == "live_discovered"
        assert source_url == "https://example.com/test-discovery-place"
    finally:
        if place_id:
            _cleanup_discovered_place(place_id)


def test_discover_live_place_persists_nothing_when_no_evidence_is_found(monkeypatch):
    # Core requirement C: insufficient evidence must not produce a fake
    # score, and must not silently persist a place with nothing behind it.
    import agent.tools as tools_module
    from ingestion import web_enrichment

    monkeypatch.setattr(web_enrichment, "search_web", lambda q: [])
    monkeypatch.setattr(tools_module, "_wikipedia_summary", lambda name: None)

    with connect() as conn:
        result = _discover_live_place(
            conn, "Nonexistent Test Place ZZZ999", "restaurant", 55.68, 12.57, None, None
        )
    assert result is None

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM places WHERE name = 'Nonexistent Test Place ZZZ999';")
        assert cur.fetchone()[0] == 0


def test_search_place_live_points_to_the_curated_name_for_a_real_re_resolved_place(monkeypatch):
    # End-to-end (mocked Nominatim only — Tier 1 re-resolution itself
    # needs no Serper/network beyond that) version of the real "Little
    # Mermaid" case: an English query that Nominatim resolves to the
    # exact same real-world entity already curated under its Danish name.
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse([{
            "name": "Den lille Havfrue", "display_name": "Den lille Havfrue, Copenhagen, Denmark",
            "lat": "55.6928661", "lon": "12.5992896", "osm_type": "node", "osm_id": 25074274,
            "category": "tourism", "type": "artwork",
        }]),
    )
    result = search_place_live.run(query="The Little Mermaid")
    assert "already in our curated database as 'Den lille Havfrue'" in result


def test_discover_live_place_degrades_gracefully_on_a_real_serper_failure(monkeypatch):
    # Core requirement D: a genuine provider failure (timeout, network
    # error, rate limit) must degrade to "insufficient evidence," not
    # crash the whole request or fabricate a score.
    import agent.tools as tools_module
    from ingestion import web_enrichment

    def _raise(*a, **k):
        raise requests.exceptions.Timeout("Serper timed out")

    monkeypatch.setattr(web_enrichment, "search_web", _raise)
    monkeypatch.setattr(tools_module, "_wikipedia_summary", lambda name: None)

    with connect() as conn:
        result = _discover_live_place(
            conn, "Serper Failure Test Place", "restaurant", 55.68, 12.57, None, None
        )
    assert result is None

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM places WHERE name = 'Serper Failure Test Place';")
        assert cur.fetchone()[0] == 0


def test_identical_repeated_tool_calls_reuse_the_cached_result(monkeypatch):
    # Core requirement L: the system must never depend on the LLM
    # voluntarily choosing not to repeat itself — a real, deterministic
    # backstop against wasted repeated work within one request.
    import agent.tools as tools_module

    tools_module.reset_tool_call_cache()
    call_count = {"n": 0}
    real_rows = tools_module._search_places_rows

    def _counting_rows(*a, **k):
        call_count["n"] += 1
        return real_rows(*a, **k)

    monkeypatch.setattr(tools_module, "_search_places_rows", _counting_rows)

    first = search_places.run(query="cozy quiet cafe")
    second = search_places.run(query="cozy quiet cafe")
    assert first == second
    assert call_count["n"] == 1, "identical repeated call should not re-run the real search"

    # A genuinely different call must NOT be served from the same cache entry.
    search_places.run(query="sushi restaurant")
    assert call_count["n"] == 2


def test_reset_tool_call_cache_clears_previous_results():
    import agent.tools as tools_module

    tools_module._tool_call_cache[("fake_tool", ())] = "stale result"
    tools_module.reset_tool_call_cache()
    assert tools_module._tool_call_cache == {}


def test_searching_the_same_new_place_twice_does_not_create_a_duplicate_row(monkeypatch):
    # Core requirement G: exact OSM identity first, then coordinate
    # proximity, before ever creating a new place row — searching the
    # same genuinely-new place twice must reuse the row the first search
    # created, not insert a second one.
    import agent.tools as tools_module
    from ingestion import web_enrichment

    monkeypatch.setattr(
        web_enrichment, "search_web",
        lambda q: [{
            "title": "Duplicate Protection Test Place - Copenhagen guide",
            "link": "https://example.com/duplicate-protection-test",
            "snippet": "A real test venue in Copenhagen with a consistently good real atmosphere and reviews.",
        }],
    )
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse([{
            "name": "Duplicate Protection Test Place",
            "display_name": "Duplicate Protection Test Place, Copenhagen, Denmark",
            "lat": "55.6900", "lon": "12.5700",
            "osm_type": "node", "osm_id": 999888777,
            "category": "amenity", "type": "restaurant",
        }]),
    )

    place_id = None
    try:
        tools_module.reset_tool_call_cache()
        first = search_place_live.run(query="Duplicate Protection Test Place")
        assert "found and added" in first

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT place_id FROM places WHERE osm_id = 'node/999888777';"
            )
            place_id = cur.fetchone()[0]

        tools_module.reset_tool_call_cache()  # isolate from the new request-level tool cache
        second = search_place_live.run(query="Duplicate Protection Test Place")
        assert "already in our curated database" in second

        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM places WHERE osm_id = 'node/999888777';")
            assert cur.fetchone()[0] == 1
    finally:
        if place_id:
            _cleanup_discovered_place(place_id)
