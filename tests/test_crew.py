"""Needs the isolated 'agent' venv (crewai) — see README's Phase 11 setup note."""

import sys
from pathlib import Path

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


class _FakeCrew:
    def __init__(self, kickoff_fn):
        self._kickoff_fn = kickoff_fn

    def kickoff(self, inputs=None):
        return self._kickoff_fn()


class _FakeCrewOutput:
    """Minimal stand-in for CrewAI's real CrewOutput — only the `.pydantic`
    attribute _extract_spec/_extract_narration actually read."""

    def __init__(self, pydantic_obj):
        self.pydantic = pydantic_obj


def test_instrument_llm_blocks_calls_past_the_configured_budget(monkeypatch):
    # The hard safety net item 2/10 in the OpenAI migration required: ONE
    # trip-plan request can never make more than MAX_LLM_CALLS_PER_REQUEST
    # real LLM calls, enforced by raising before the (N+1)th call is ever
    # attempted — not just counted after the fact.
    monkeypatch.setattr(crew_module, "MAX_LLM_CALLS_PER_REQUEST", 2)
    crew_module._llm_call_count.set(0)

    class _FakeLLM:
        def call(self, *a, **k):
            return "ok"

    llm = crew_module._instrument_llm(_FakeLLM())
    assert llm.call() == "ok"
    assert llm.call() == "ok"
    try:
        llm.call()
        assert False, "3rd call should have been blocked at MAX_LLM_CALLS_PER_REQUEST=2"
    except crew_module.TripPlannerLLMUnavailable:
        pass


def test_instrument_llm_caps_output_tokens_via_build_llm(monkeypatch):
    # OPENAI_MAX_OUTPUT_TOKENS must actually reach the LLM config passed to
    # crewai — not just exist as an env var nobody reads.
    monkeypatch.setattr(crew_module, "OPENAI_MAX_OUTPUT_TOKENS", 42)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-this-test-only")
    cfg = crew_module.build_llm()
    assert cfg["max_tokens"] == 42
    assert cfg["max_retries"] == 0
    assert "groq" not in cfg["model"].lower()


def test_plan_trip_never_retries_an_llm_unavailable_hit(monkeypatch):
    # A retry here means re-running the ENTIRE crew, not just the failed
    # call — on a small, fixed OpenAI credit budget, an automatic retry
    # gambles a full run's worth of tokens rather than just failing fast
    # into api/main.py's deterministic fallback. TripPlannerLLMUnavailable
    # is what _instrument_llm's llm.call() wrapper (agent/crew.py) actually
    # raises for both a real OpenAI failure and a MAX_LLM_CALLS_PER_REQUEST
    # hit — this fake crew simulates the intent-extraction step itself
    # failing. Confirms build_intent_crew is called exactly once, with no
    # retry, and the Concierge crew is never reached.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-this-test-only")
    monkeypatch.setattr(crew_module, "_get_exact_cache", lambda *a, **k: None)
    monkeypatch.setattr(crew_module, "_save_cache", lambda *a, **k: None)

    calls = {"intent": 0, "concierge": 0}

    def kickoff_fn():
        calls["intent"] += 1
        raise crew_module.TripPlannerLLMUnavailable("rate limited")

    monkeypatch.setattr(crew_module, "build_intent_crew", lambda llm: _FakeCrew(kickoff_fn))
    monkeypatch.setattr(
        crew_module, "build_concierge_crew",
        lambda llm: _FakeCrew(lambda: calls.__setitem__("concierge", calls["concierge"] + 1)),
    )

    try:
        crew_module.plan_trip("test request", "2026-01-01")
        assert False, "expected TripPlannerLLMUnavailable to propagate"
    except crew_module.TripPlannerLLMUnavailable:
        pass
    assert calls["intent"] == 1
    assert calls["concierge"] == 0


def test_narration_guard_catches_the_exact_test_b_bare_near_wording():
    # Real, live-observed regression: "conveniently located near several
    # cozy cafes" slipped past the phrase-list guard because the bare word
    # "near" wasn't in it (deliberately — see the next test). The cafes in
    # this exact real request were actually 2.87-5.04 km away.
    results = [
        {"name": "Den lille Havfrue", "near_place": None},
        {"name": "Switch Coffee", "near_place": None},
        {"name": "Roast Coffee", "near_place": None},
    ]
    narration = crew_module.ConciergeNarration(
        weather_summary="sunny",
        overall_note=(
            "Visiting the Little Mermaid is a must when in Copenhagen, and it's conveniently "
            "located near several cozy cafes for a delightful coffee afterward."
        ),
        place_narrations=[],
    )

    fixed = crew_module._enforce_narration_matches_deterministic_relationships(results, narration)

    assert "near several cozy cafes" not in fixed.overall_note.lower()


def test_narration_guard_leaves_an_unrelated_geographic_description_untouched():
    # The exact reason bare "near" isn't simply added to the phrase list:
    # "near the lakes" is a real, true, unrelated geographic aside (Hotel
    # Nora's actual location relative to Copenhagen's Lakes, not a claim
    # about distance to another recommended place) and must survive.
    results = [{"name": "Hotel Nora", "near_place": None}]
    narration = crew_module.ConciergeNarration(
        weather_summary="sunny",
        overall_note="Hotel Nora offers a prime location near the lakes and easy access to shopping.",
        place_narrations=[
            crew_module.PlaceNarration(
                name="Hotel Nora",
                why_recommended="A prime location near the lakes with a great breakfast buffet.",
            ),
        ],
    )

    fixed = crew_module._enforce_narration_matches_deterministic_relationships(results, narration)

    assert fixed.overall_note == "Hotel Nora offers a prime location near the lakes and easy access to shopping."
    assert fixed.place_narrations[0].why_recommended == "A prime location near the lakes with a great breakfast buffet."


def test_narration_guard_strips_a_false_proximity_claim_for_a_non_near_place():
    # Real, live-observed gap: the Concierge's free-text prose sometimes
    # used casual proximity language ("nearby", "a short walk") for a
    # place whose real, computed near_place was None. Must be caught and
    # replaced deterministically, without touching unrelated true content.
    results = [
        {"name": "Cafe A", "near_place": None},
        {"name": "Den lille Havfrue", "near_place": None},
    ]
    narration = crew_module.ConciergeNarration(
        weather_summary="sunny",
        overall_note="Visit the Mermaid, then unwind at one of the cozy cafes nearby. It will be sunny.",
        place_narrations=[
            crew_module.PlaceNarration(name="Cafe A", why_recommended="A short walk from the Mermaid, this cafe is lovely."),
            crew_module.PlaceNarration(name="Den lille Havfrue", why_recommended="An iconic statue worth seeing."),
        ],
    )

    fixed = crew_module._enforce_narration_matches_deterministic_relationships(results, narration)

    assert "nearby" not in fixed.overall_note.lower()
    assert "It will be sunny." in fixed.overall_note
    assert "short walk" not in fixed.place_narrations[0].why_recommended.lower()
    assert fixed.place_narrations[1].why_recommended == "An iconic statue worth seeing."  # untouched, no claim to begin with


def test_narration_guard_never_touches_a_true_proximity_claim_for_a_real_near_place():
    results = [{"name": "Cafe B", "near_place": "Den lille Havfrue"}]
    narration = crew_module.ConciergeNarration(
        weather_summary="sunny",
        overall_note="Cafe B is nearby the statue, a lovely short walk away.",
        place_narrations=[
            crew_module.PlaceNarration(name="Cafe B", why_recommended="A short walk from the Mermaid, this cafe is lovely."),
        ],
    )

    fixed = crew_module._enforce_narration_matches_deterministic_relationships(results, narration)

    assert fixed.overall_note == "Cafe B is nearby the statue, a lovely short walk away."
    assert fixed.place_narrations[0].why_recommended == "A short walk from the Mermaid, this cafe is lovely."


def test_run_structured_trip_plan_synthesizes_from_real_results_when_only_the_concierge_fails(monkeypatch):
    # The new failure mode this architecture introduces: intent extraction
    # succeeds and execute_trip_specification() already produced real,
    # scored places, but the Concierge's final narration call fails. Must
    # not lose that real work — _synthesize_without_concierge should
    # produce an honest response directly from it, zero extra LLM cost.
    from agent.intent import ItineraryPart, TripSpecification

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-this-test-only")
    spec = TripSpecification(
        parts=[ItineraryPart(sequence_index=0, query="Test Place", named_place=True, relation="primary")]
    )
    monkeypatch.setattr(
        crew_module, "build_intent_crew",
        lambda llm: _FakeCrew(lambda: _FakeCrewOutput(spec)),
    )

    fake_results = [{
        "name": "Test Place", "category": "cafe", "neighborhood": "Indre By",
        "opening_hours": None, "recommendation_confidence": 80, "recommendation_label": "recommended",
        "vibe_cluster": None, "summary": None, "sources": [],
        "distance_km": 1.2, "walk_minutes": 15, "bike_minutes": 5, "travel_note": None,
        "near_place": None, "near_distance_km": None,
        "sequence_index": 0, "relation": "primary", "source": "database", "query": "Test Place",
    }]
    monkeypatch.setattr(crew_module, "execute_trip_specification", lambda spec: fake_results)

    def _raise_unavailable():
        raise crew_module.TripPlannerLLMUnavailable("rate limited on the 2nd call")

    monkeypatch.setattr(crew_module, "build_concierge_crew", lambda llm: _FakeCrew(_raise_unavailable))
    monkeypatch.setattr(crew_module, "_weather_structured", lambda target_date, category="": {"ok": False, "error_kind": "provider_unavailable", "date": target_date})

    result = crew_module._run_structured_trip_plan("see Test Place", "2026-01-01", "")

    assert [p["name"] for p in result["places"]] == ["Test Place"]
    assert result["places"][0]["recommendation_confidence"] == 80
    assert result["places"][0]["distance_km"] == 1.2
    assert "temporary limit" in result["overall_note"]


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


def test_compound_deterministic_places_gives_the_primary_place_start_distance_but_not_secondary(monkeypatch):
    # Real bug this guards: Vanløse -> Little Mermaid -> Cafe was showing
    # the cafe's distance FROM VANLØSE ("6.8 km from your start") instead
    # of from the Little Mermaid, because a real start_coords was being
    # threaded into the secondary place's kwargs too. The primary place
    # SHOULD still get a real distance-from-start (that hop genuinely is
    # start -> primary) — only the secondary "near X" place must not.
    split = crew_module._split_compound_request(
        "wanna see little mermaid and after go to a nearby restaurant"
    )
    assert split is not None

    vanlose_coords = (55.6761, 12.5306)  # real, approximate Vanløse coordinates
    with connect() as conn, conn.cursor() as cur:
        places = crew_module._compound_deterministic_places(cur, conn, split, start_coords=vanlose_coords)

    primary, secondary = places[0], places[1:]
    assert primary.name == "Den lille Havfrue"
    assert primary.distance_km is not None, "the primary place's own distance from the real start must still be set"
    assert len(secondary) >= 1
    for p in secondary:
        assert p.near_place == "Den lille Havfrue"
        assert p.near_distance_km is not None
        assert p.distance_km is None, (
            f"{p.name} must not carry a start-relative distance_km — its relevant "
            "distance is near_distance_km, from the primary place, not from Vanløse"
        )


def test_reconcile_near_relationships_fixes_a_wrong_reference_distance():
    # The real fix for the reported production bug: even if the Concierge
    # (an LLM) wrote a start-relative distance_km for a place that's
    # actually near another recommended place, this backend step
    # overwrites it using ONLY the Scout's real, cached search_places_near
    # call — never trusting anything the LLM itself wrote.
    tools_module.reset_tool_call_cache()
    tools_module._tool_call_cache[
        ("search_places_near", (("anchor_place", "Den lille Havfrue"), ("category", "restaurant")))
    ] = "irrelevant cached text — only the kwargs are read"

    with connect() as conn, conn.cursor() as cur:
        rows = tools_module._places_near(
            cur, 55.6928661, 12.5992896, category="restaurant", exclude_name="Den lille Havfrue", limit=1,
        )
    assert rows, "expected at least one real restaurant near the Little Mermaid in the live database"
    nearest_name, real_distance = rows[0]["name"], round(rows[0]["distance_km"], 2)

    result = {
        "places": [
            {
                "name": nearest_name,
                # Wrong-reference values an LLM might have written — exactly
                # the reported bug ("6.8 km from your start" for the cafe).
                "distance_km": 6.8, "walk_minutes": 85, "bike_minutes": 27,
                "travel_note": "consider transit",
                "near_place": None, "near_distance_km": None,
            }
        ]
    }
    fixed = crew_module._reconcile_near_relationships(result)
    place = fixed["places"][0]
    assert place["near_place"] == "Den lille Havfrue"
    assert place["near_distance_km"] == real_distance
    assert place["distance_km"] is None
    assert place["walk_minutes"] is None
    assert place["bike_minutes"] is None
    assert place["travel_note"] is None
    tools_module.reset_tool_call_cache()


def test_reconcile_near_relationships_matches_a_shortened_llm_written_name():
    # Real gap found live: the Concierge's final answer wrote the real
    # place's name shortened ("Terminalen kaffebar" for the database's
    # real "Terminalen kaffebar - Seaside Toldboden") — an exact-string
    # lookup would silently miss it, leaving a stale distance_km
    # uncorrected. Must still match via substring.
    tools_module.reset_tool_call_cache()
    tools_module._tool_call_cache[
        ("search_places_near", (("anchor_place", "Den lille Havfrue"), ("category", "cafe")))
    ] = "irrelevant cached text — only the kwargs are read"

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT name FROM places WHERE name ILIKE %s;", ("%terminalen%",))
        real_name = cur.fetchone()[0]
    assert real_name == "Terminalen kaffebar - Seaside Toldboden"
    shortened_name = "Terminalen kaffebar"

    result = {"places": [{"name": shortened_name, "distance_km": 6.8, "walk_minutes": 82, "bike_minutes": 27,
                           "travel_note": "consider transit", "near_place": None, "near_distance_km": None}]}
    fixed = crew_module._reconcile_near_relationships(result)
    place = fixed["places"][0]
    assert place["near_place"] == "Den lille Havfrue"
    assert place["near_distance_km"] is not None
    assert place["distance_km"] is None, "the shortened name must still match and clear the stale start-distance"
    tools_module.reset_tool_call_cache()


def test_reconcile_near_relationships_is_a_noop_for_a_plain_single_destination_request():
    # Existing single-destination behavior must stay exactly as-is: if the
    # Scout never called search_places_near at all this request, nothing
    # should be touched or overwritten.
    tools_module.reset_tool_call_cache()
    result = {
        "places": [
            {"name": "Torvehallerne", "distance_km": 3.1, "walk_minutes": 38, "bike_minutes": 13,
             "travel_note": None, "near_place": None, "near_distance_km": None}
        ]
    }
    unchanged = crew_module._reconcile_near_relationships(dict(result))
    assert unchanged == result


def test_recompute_travel_never_overwrites_a_place_that_already_has_a_near_place():
    # A cache-reuse request with a DIFFERENT start location must not
    # clobber a secondary place's real near_distance_km with a new
    # start-relative distance_km — its relevant reference never changes
    # just because the traveler's own starting point did.
    result = {
        "places": [
            {"name": "Den lille Havfrue", "distance_km": None, "near_place": None, "near_distance_km": None},
            {"name": "Nonna regina", "distance_km": None, "walk_minutes": None, "bike_minutes": None,
             "travel_note": None, "near_place": "Den lille Havfrue", "near_distance_km": 0.30},
        ]
    }
    updated = crew_module._recompute_travel(result, 55.6761, 12.5306, "Vanløse")
    primary, secondary = updated["places"]
    assert primary["distance_km"] is not None, "the primary place should still get a real distance from the new start"
    assert secondary["distance_km"] is None, "a place with near_place set must not gain a start-relative distance"
    assert secondary["near_place"] == "Den lille Havfrue"
    assert secondary["near_distance_km"] == 0.30


def test_ensure_start_distance_backfills_a_place_the_concierge_forgot(monkeypatch):
    # Real gap found live: the Concierge's own batched travel_time_estimate
    # call sometimes only named the secondary places, leaving the PRIMARY
    # place's distance_km null even though a real starting point was
    # given. This must be backfilled deterministically, not left missing.
    tools_module.set_trip_start(55.6761, 12.5306, "Vanlose")
    try:
        result = {
            "places": [
                {"name": "Den lille Havfrue", "distance_km": None, "walk_minutes": None,
                 "bike_minutes": None, "travel_note": None, "near_place": None, "near_distance_km": None},
                {"name": "Nonna regina", "distance_km": None, "walk_minutes": None, "bike_minutes": None,
                 "travel_note": None, "near_place": "Den lille Havfrue", "near_distance_km": 0.30},
            ]
        }
        updated = crew_module._ensure_start_distance(result)
    finally:
        tools_module.set_trip_start(None, None, None)

    primary, secondary = updated["places"]
    assert primary["distance_km"] is not None, "a missing primary distance must be backfilled from the real start"
    assert secondary["distance_km"] is None, "a place with near_place set must never get a backfilled start distance"
    assert secondary["near_distance_km"] == 0.30, "an already-set near_distance_km must be left untouched"


def test_ensure_start_distance_is_a_noop_without_a_real_starting_point():
    tools_module.set_trip_start(None, None, None)
    result = {"places": [{"name": "Den lille Havfrue", "distance_km": None, "near_place": None, "near_distance_km": None}]}
    unchanged = crew_module._ensure_start_distance(dict(result))
    assert unchanged == result


def test_ensure_start_distance_never_overwrites_an_already_set_distance():
    tools_module.set_trip_start(55.6761, 12.5306, "Vanlose")
    try:
        result = {"places": [{"name": "Den lille Havfrue", "distance_km": 1.23, "walk_minutes": 15,
                               "bike_minutes": 5, "travel_note": None, "near_place": None, "near_distance_km": None}]}
        updated = crew_module._ensure_start_distance(dict(result))
    finally:
        tools_module.set_trip_start(None, None, None)
    assert updated["places"][0]["distance_km"] == 1.23, "an already-set real distance must never be recomputed/overwritten"
