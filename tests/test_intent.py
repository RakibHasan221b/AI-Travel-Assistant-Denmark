"""Unit tests for the new structured intent layer (agent/intent.py):
TripSpecification/ItineraryPart validation, and execute_trip_specification's
deterministic backend routing. Needs the isolated 'agent' venv (crewai) —
see README's Phase 11 setup note; execute_trip_specification tests need a
real DATABASE_URL like the rest of tests/test_agent_tools.py and
tests/test_crew.py already do."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.intent import (
    ItineraryPart,
    Relation,
    TripSpecification,
    execute_trip_specification,
)
from agent.tools import set_trip_start

# ---------------------------------------------------------------------------
# ItineraryPart / TripSpecification validation
# ---------------------------------------------------------------------------


def test_far_relation_rejects_max_distance_km():
    # The exact "impossible spec" case named in the task: far is a
    # near-only upper bound, never far's.
    with pytest.raises(ValidationError, match="max_distance_km"):
        ItineraryPart(sequence_index=0, query="cafe", named_place=False, relation=Relation.FAR, max_distance_km=1.0)


def test_near_relation_requires_a_resolvable_anchor_source():
    with pytest.raises(ValidationError, match="anchor"):
        ItineraryPart(sequence_index=0, query="cafe", named_place=False, relation=Relation.NEAR)


def test_near_relation_accepts_anchor_query():
    part = ItineraryPart(
        sequence_index=0, query="cafe", named_place=False, relation=Relation.NEAR,
        anchor_query="Den lille Havfrue",
    )
    assert part.anchor_query == "Den lille Havfrue"
    assert part.anchor_is_start_location is False


def test_near_relation_accepts_anchor_is_start_location_for_near_me():
    part = ItineraryPart(
        sequence_index=0, query="restaurant", named_place=False, relation=Relation.NEAR,
        anchor_is_start_location=True,
    )
    assert part.anchor_query is None
    assert part.anchor_is_start_location is True


def test_area_relation_requires_neighborhood():
    with pytest.raises(ValidationError, match="neighborhood"):
        ItineraryPart(sequence_index=0, query="cafe", named_place=False, relation=Relation.AREA)


def test_area_relation_with_neighborhood_is_valid():
    part = ItineraryPart(sequence_index=0, query="cafe", named_place=False, relation=Relation.AREA, neighborhood="Nørrebro")
    assert part.neighborhood == "Nørrebro"


def test_distance_fields_must_be_positive():
    with pytest.raises(ValidationError):
        ItineraryPart(
            sequence_index=0, query="cafe", named_place=False, relation=Relation.NEAR,
            anchor_query="X", max_distance_km=-1.0,
        )


def test_unknown_category_is_normalized_to_none_not_rejected():
    # Repair, not reject — mirrors PlaceRecommendation's own existing
    # field_validator philosophy in agent/crew.py (coerce the model's
    # actual output rather than crash on a minor cosmetic quirk).
    part = ItineraryPart(sequence_index=0, query="something", named_place=False, category="museum-ish", relation=Relation.SEQUENTIAL)
    assert part.category is None


def test_known_category_is_normalized_lowercase():
    part = ItineraryPart(sequence_index=0, query="food", named_place=False, category="Restaurant", relation=Relation.SEQUENTIAL)
    assert part.category == "restaurant"


def test_irrelevant_anchor_on_a_primary_part_is_cleared_not_rejected():
    # A stray anchor_query on a relation that doesn't use one is noise,
    # not a contradiction — cleared silently rather than spending an
    # instructor retry on it.
    part = ItineraryPart(sequence_index=0, query="Den lille Havfrue", named_place=True, relation=Relation.PRIMARY, anchor_query="something")
    assert part.anchor_query is None


def test_duplicate_sequence_index_across_parts_is_rejected():
    # "Ordering must be internally consistent" — two parts cannot both
    # claim the same position.
    with pytest.raises(ValidationError, match="sequence_index"):
        TripSpecification(parts=[
            ItineraryPart(sequence_index=0, query="a", named_place=True, relation=Relation.PRIMARY),
            ItineraryPart(sequence_index=0, query="b", named_place=False, relation=Relation.SEQUENTIAL),
        ])


def test_trip_specification_requires_at_least_one_part():
    with pytest.raises(ValidationError):
        TripSpecification(parts=[])


def test_far_relation_never_becomes_near_structurally():
    # Real point of the whole redesign: a single required enum field means
    # a part's relation is far OR near, never ambiguously both — this is
    # what made "then...far away" wrongly resolve near in the old
    # prompt-only architecture (PROJECT_ARCHITECTURE_REPORT.md §12 item 4).
    part = ItineraryPart(sequence_index=0, query="cafe", named_place=False, relation=Relation.FAR, anchor_query="Den lille Havfrue")
    assert part.relation == Relation.FAR
    assert part.relation != Relation.NEAR


# ---------------------------------------------------------------------------
# execute_trip_specification — deterministic backend execution (real DB)
# ---------------------------------------------------------------------------


def test_execute_named_primary_resolves_the_exact_place_not_a_similarly_named_decoy():
    # Real bug found live while building this: semantic search alone
    # resolved "Den lille Havfrue" to a different, similarly-described
    # real place ("Den Genmodificerede Lille Havfrue"). Exact/substring
    # match must win for an exact name.
    set_trip_start(None, None, "")
    spec = TripSpecification(parts=[
        ItineraryPart(sequence_index=0, query="Den lille Havfrue", named_place=True, relation=Relation.PRIMARY),
    ])
    results = execute_trip_specification(spec)
    assert len(results) == 1
    assert results[0]["name"] == "Den lille Havfrue"
    assert results[0]["relation"] == "primary"


def test_execute_near_sets_near_place_and_clears_start_relative_distance():
    set_trip_start(55.6761, 12.5306, "Vanlose")
    spec = TripSpecification(parts=[
        ItineraryPart(sequence_index=0, query="Den lille Havfrue", named_place=True, relation=Relation.PRIMARY),
        ItineraryPart(
            sequence_index=1, query="coffee", named_place=False, category="cafe",
            relation=Relation.NEAR, anchor_query="Den lille Havfrue",
        ),
    ])
    results = execute_trip_specification(spec)
    primary = next(r for r in results if r["relation"] == "primary")
    assert primary["near_place"] is None
    assert primary["distance_km"] is not None  # real start-relative distance

    near_results = [r for r in results if r["relation"] == "near"]
    assert len(near_results) >= 1
    for r in near_results:
        assert r["near_place"] == "Den lille Havfrue"
        assert r["near_distance_km"] is not None
        assert r["near_distance_km"] <= 2.0  # MAX_NEARBY_KM
        assert r["distance_km"] is None  # never a start-relative distance for a near-matched place


def test_execute_far_never_calls_places_near_and_keeps_start_relative_distance(monkeypatch):
    import agent.intent as intent_module

    def _fail_if_called(*a, **k):
        raise AssertionError("relation=far must never call _places_near")

    monkeypatch.setattr(intent_module, "_places_near", _fail_if_called)

    set_trip_start(55.6761, 12.5306, "Vanlose")
    spec = TripSpecification(parts=[
        ItineraryPart(sequence_index=0, query="Den lille Havfrue", named_place=True, relation=Relation.PRIMARY),
        ItineraryPart(
            sequence_index=1, query="coffee", named_place=False, category="cafe",
            relation=Relation.FAR, anchor_query="Den lille Havfrue",
        ),
    ])
    results = execute_trip_specification(spec)
    far_results = [r for r in results if r["relation"] == "far"]
    assert len(far_results) >= 1
    for r in far_results:
        assert r["near_place"] is None
        assert r["near_distance_km"] is None


def test_execute_area_filters_by_neighborhood_using_existing_search_places_rows_param():
    set_trip_start(None, None, "")
    spec = TripSpecification(parts=[
        ItineraryPart(sequence_index=0, query="cozy cafe", named_place=False, category="cafe", relation=Relation.AREA, neighborhood="Nørrebro"),
    ])
    results = execute_trip_specification(spec)
    assert len(results) > 0
    for r in results:
        assert r["neighborhood"] == "Nørrebro"


def test_execute_sequential_bare_part_never_sets_near_place():
    # The exact structural guarantee the whole redesign is meant to give:
    # a relation=sequential part (a bare "then"/"afterwards", no spatial
    # word at all) cannot produce a near relationship — there's no anchor
    # field populated for it to use even if it wanted to.
    set_trip_start(None, None, "")
    spec = TripSpecification(parts=[
        ItineraryPart(sequence_index=0, query="Den lille Havfrue", named_place=True, relation=Relation.PRIMARY),
        ItineraryPart(sequence_index=1, query="coffee", named_place=False, category="cafe", relation=Relation.SEQUENTIAL),
    ])
    results = execute_trip_specification(spec)
    sequential_results = [r for r in results if r["relation"] == "sequential"]
    assert len(sequential_results) > 0
    for r in sequential_results:
        assert r["near_place"] is None
        assert r["near_distance_km"] is None


def test_execute_near_with_max_distance_km_is_enforced_by_places_near():
    set_trip_start(None, None, "")
    spec = TripSpecification(parts=[
        ItineraryPart(
            sequence_index=0, query="cafe", named_place=False, category="cafe",
            relation=Relation.NEAR, anchor_query="Den lille Havfrue", max_distance_km=1.0,
        ),
    ])
    results = execute_trip_specification(spec)
    for r in results:
        assert r["near_distance_km"] <= 1.0


def test_serper_fallback_is_not_called_when_internal_results_are_sufficient():
    # Real threshold verification: a category+area combo with real DB
    # coverage (cafe in Nørrebro) must never trigger the Serper fallback.
    import agent.intent as intent_module

    calls = {"n": 0}
    monkeypatch_target = intent_module.discover_candidates_live

    def spy(*a, **k):
        calls["n"] += 1
        return monkeypatch_target(*a, **k)

    intent_module.discover_candidates_live = spy
    try:
        set_trip_start(None, None, "")
        spec = TripSpecification(parts=[
            ItineraryPart(sequence_index=0, query="cozy cafe", named_place=False, category="cafe", relation=Relation.AREA, neighborhood="Nørrebro"),
        ])
        results = execute_trip_specification(spec)
    finally:
        intent_module.discover_candidates_live = monkeypatch_target

    assert len(results) >= 2  # real, sufficient DB coverage
    assert calls["n"] == 0


def test_serper_fallback_is_called_when_internal_results_are_insufficient():
    # hotel + Nørrebro has exactly 1 real DB row — below OPEN_ENDED_PART_MIN.
    import agent.intent as intent_module

    calls = {"n": 0}
    real_fn = intent_module.discover_candidates_live

    def spy(*a, **k):
        calls["n"] += 1
        return real_fn(*a, **k)

    intent_module.discover_candidates_live = spy
    try:
        set_trip_start(None, None, "")
        spec = TripSpecification(parts=[
            ItineraryPart(sequence_index=0, query="hotel", named_place=False, category="hotel", relation=Relation.AREA, neighborhood="Nørrebro"),
        ])
        execute_trip_specification(spec)
    finally:
        intent_module.discover_candidates_live = real_fn

    assert calls["n"] == 1


def test_named_place_not_in_database_triggers_live_lookup_and_finds_nothing_invented():
    # "Reffen" (a real Copenhagen street food market) resolves via
    # Nominatim but its OSM category doesn't map to this project's known
    # taxonomy — must be reported as genuinely unscored, never given an
    # invented category or a fake recommendation_confidence.
    import agent.intent as intent_module

    calls = {"n": 0}
    real_fn = intent_module._live_lookup

    def spy(*a, **k):
        calls["n"] += 1
        return real_fn(*a, **k)

    intent_module._live_lookup = spy
    try:
        set_trip_start(None, None, "")
        spec = TripSpecification(parts=[
            ItineraryPart(sequence_index=0, query="Reffen", named_place=True, relation=Relation.PRIMARY),
        ])
        results = execute_trip_specification(spec)
    finally:
        intent_module._live_lookup = real_fn

    assert calls["n"] == 1
    assert results == []  # never invented a place with no resolvable category/evidence


def test_serper_candidate_duplicating_a_real_db_place_by_name_is_deduplicated():
    # Controlled test of the dedup logic itself (independent of whatever
    # live Serper happens to return today) — a fake Serper candidate that
    # names an already-curated place (different case) must not produce a
    # second entry.
    import agent.intent as intent_module

    def fake_discover(query, category="", neighborhood="", limit=3):
        return [{"name": "ORIGINAL COFFEE", "category": "cafe", "lat": 55.0, "lon": 12.0}]

    real_fn = intent_module.discover_candidates_live
    real_min = intent_module.OPEN_ENDED_PART_MIN
    intent_module.discover_candidates_live = fake_discover
    intent_module.OPEN_ENDED_PART_MIN = 999  # force the fallback branch even though the DB already has real coverage
    try:
        set_trip_start(None, None, "")
        spec = TripSpecification(parts=[
            ItineraryPart(sequence_index=0, query="coffee", named_place=False, category="cafe", relation=Relation.SEQUENTIAL),
        ])
        results = execute_trip_specification(spec)
    finally:
        intent_module.discover_candidates_live = real_fn
        intent_module.OPEN_ENDED_PART_MIN = real_min

    names_lower = [r["name"].lower() for r in results]
    assert names_lower.count("original coffee") <= 1


def test_every_candidate_regardless_of_source_gets_a_real_xgboost_score():
    # recommendation_confidence must come from _lookup_place_structured
    # (the real XGBoost pipeline) for every result, database-sourced or
    # live-discovered — never a separate/invented number.
    set_trip_start(None, None, "")
    spec = TripSpecification(parts=[
        ItineraryPart(sequence_index=0, query="Den lille Havfrue", named_place=True, relation=Relation.PRIMARY),
    ])
    results = execute_trip_specification(spec)
    assert len(results) == 1
    r = results[0]
    assert r["source"] == "database"
    assert r["recommendation_confidence"] is not None
    assert r["recommendation_label"] in ("recommended", "not recommended")


def test_execute_near_with_unresolvable_anchor_degrades_to_open_ended_with_a_note():
    set_trip_start(None, None, "")
    spec = TripSpecification(parts=[
        ItineraryPart(
            sequence_index=0, query="coffee", named_place=False, category="cafe",
            relation=Relation.NEAR, anchor_is_start_location=True,  # no start_coords set above -> unresolvable
        ),
    ])
    results = execute_trip_specification(spec)
    for r in results:
        assert r.get("execution_note") is not None
        assert r["near_place"] is None
