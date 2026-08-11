"""Structured intent layer — the boundary move requested after
PROJECT_ARCHITECTURE_REPORT.md's investigation found the Trip Planner's
near/far/sequential classification lived entirely in Scout prompt prose,
with zero structural backstop (see that report's §3, §9, §15). Confirmed
live: a bare "...and then have coffee" (no spatial word at all) still
triggered search_places_near — proof that a single LLM reasoning pass
over prose cannot reliably enforce a decision boundary, however
explicitly worded.

This module does NOT change how places are searched, scored, or
distanced — it changes WHO decides which search to run. TripSpecification
is what the Scout now produces (via CrewAI's existing output_pydantic/
instructor mechanism, the same one already proven on concierge_task);
execute_trip_specification() is the deterministic backend that maps a
validated spec onto the *existing* agent/tools.py functions
(_search_places_rows, _places_near, _resolve_place, _lookup_place_structured)
— never new search/scoring logic, just backend-owned routing instead of
LLM-owned tool choice.

Deliberately depends only on agent.tools, never agent.crew (which depends
on this module instead) — avoids a circular import, and keeps this
module callable/testable without spinning up CrewAI at all.
"""

import logging
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.tools import (
    MAX_NEARBY_KM,
    _live_lookup,
    _lookup_place_structured,
    _places_near,
    _resolve_place,
    _resolve_place_exact,
    _search_places_rows,
    connect,
    discover_candidates_live,
    get_trip_start,
    haversine_km,
    travel_fields,
)

log = logging.getLogger("intent")

# The project's existing category vocabulary — see db/schema.sql's
# `category` column comment and every search_* tool's own docstring in
# agent/tools.py (all consistently list these five). Not redefined here
# as a source of truth, just mirrored, so an LLM-supplied category that
# isn't one of these gets normalized to null rather than silently
# admitted as a stray, unfilterable string.
KNOWN_CATEGORIES = {"restaurant", "cafe", "hotel", "landmark", "bar"}

# How many genuinely relevant internal candidates a part needs before
# execute_trip_specification() treats the internal database as
# "sufficient" and skips the Serper candidate-discovery fallback (see
# discover_candidates_live in agent/tools.py). Real, chosen thresholds,
# not arbitrary:
#  - NAMED_PART_MIN=1: a named place either resolves to one real thing or
#    it doesn't — there's no concept of "a few named-place candidates."
#  - OPEN_ENDED_PART_MIN=2: mirrors the old Scout prompt's own guidance
#    ("return up to 3-5 candidates, fewer if fewer genuinely fit") — 2 is
#    the point below which a traveler is being offered close to nothing.
#  - NEAR_PART_MIN=1: a near-search's candidate pool is inherently small
#    (capped at MAX_NEARBY_KM), so even one genuine nearby match beats
#    none, unlike a city-wide open-ended search.
NAMED_PART_MIN = 1
OPEN_ENDED_PART_MIN = 2
NEAR_PART_MIN = 1


class Relation(str, Enum):
    """A part's spatial relationship to its anchor (or lack of one) —
    kept deliberately separate from ordering (see ItineraryPart.
    sequence_index below): 'then'/'before'/'after' are about WHEN, never
    WHERE. Conflating the two inside one signal is exactly what caused
    the prompt-only Scout to misclassify 'then'/'afterwards' as proximity
    (PROJECT_ARCHITECTURE_REPORT.md §12 items 4-5) — giving sequence its
    own field removes the temptation entirely, it's no longer possible
    for a sequence word to even be *the same field* as a proximity word."""

    PRIMARY = "primary"        # the reference part itself — no anchor, no spatial relation to anything else
    NEAR = "near"               # explicit proximity to anchor_query, or to the traveler's own start location
    FAR = "far"                  # explicit distance/far-away wording — never search near an anchor for this part
    SEQUENTIAL = "sequential"     # only an order is stated ("then", "afterwards") — no spatial relation at all
    AREA = "area"                  # an explicit neighborhood/area constraint, no anchor place needed


class ItineraryPart(BaseModel):
    sequence_index: int = Field(
        ge=0,
        description="0-based position in the order the traveler wants to visit this part. This is the "
        "ONLY field 'before'/'after'/'then'/'first' should ever affect — never relation.",
    )
    query: str = Field(
        min_length=1,
        description="The traveler's own wording for this part — a named place ('Den lille Havfrue') or "
        "an open-ended category/vibe description ('a cozy cafe').",
    )
    named_place: bool = Field(
        description="True if the traveler named ONE specific real place for this part; False if it's "
        "open-ended (a category or vibe with no specific place named)."
    )
    category: str | None = Field(
        None,
        description=f"One of {sorted(KNOWN_CATEGORIES)} if implied by the request, else null — never "
        "invent a category the request doesn't imply.",
    )
    relation: Relation = Field(
        description="This part's spatial relationship, decided using ONLY explicit wording in the "
        "request — a sequence word ('then', 'afterwards') alone is never enough to justify near or far."
    )
    anchor_query: str | None = Field(
        None,
        description="The traveler's own wording naming the reference place, for relation=near/far only. "
        "Null otherwise, and null when anchor_is_start_location is true.",
    )
    anchor_is_start_location: bool = Field(
        False,
        description="True only when the traveler referred to their OWN starting point as the anchor "
        "(e.g. 'near me', 'close to where I'm starting') rather than another named place.",
    )
    max_distance_km: float | None = Field(
        None, gt=0,
        description="Only meaningful for relation=near: an explicit upper bound, e.g. 'within 1 km' -> "
        "1.0, 'walking distance' -> about 1.5. Null if the traveler gave no number.",
    )
    min_distance_km: float | None = Field(
        None, gt=0,
        description="Only meaningful for relation=far: an explicit lower bound, e.g. 'at least 2 km "
        "away' -> 2.0. Null if the traveler gave no number — most 'far away' requests give none, "
        "which is fine and expected.",
    )
    neighborhood: str | None = Field(
        None, description="An explicit neighborhood/area name, required when relation=area (e.g. 'Nørrebro')."
    )

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v):
        if v is None:
            return None
        v = str(v).strip().lower()
        return v if v in KNOWN_CATEGORIES else None

    @field_validator("query", "anchor_query", "neighborhood", mode="before")
    @classmethod
    def _blank_to_none_or_strip(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @model_validator(mode="after")
    def _check_relation_consistency(self):
        # Genuinely impossible/inconsistent combinations — raising here
        # (inside a CrewAI output_pydantic task, before any tool call or
        # DB query has happened) lets instructor's own existing
        # retry-on-validation-error mechanism ask the LLM to fix it, at
        # near-zero cost — a real, mechanism-backed way to reduce
        # misclassification, not just a data-repair shim.
        if self.relation == Relation.FAR and self.max_distance_km is not None:
            raise ValueError(
                "relation=far cannot carry max_distance_km (that's a near-only upper bound) — "
                "if a minimum distance was actually meant, use min_distance_km instead"
            )
        if self.relation == Relation.NEAR and not self.anchor_query and not self.anchor_is_start_location:
            raise ValueError(
                "relation=near requires anchor_query (a named reference place) or "
                "anchor_is_start_location=true (for 'near me') — a near relationship with no "
                "anchor at all is not resolvable"
            )
        if self.relation == Relation.AREA and not self.neighborhood:
            raise ValueError("relation=area requires neighborhood to be set")

        # Below: fields that are merely IRRELEVANT for this relation, not
        # contradictory — cleared rather than rejected, matching this
        # codebase's existing repair-the-actual-output philosophy (see
        # PlaceRecommendation's own field_validators in agent/crew.py).
        # Not worth spending an instructor retry on.
        if self.relation not in (Relation.NEAR, Relation.FAR):
            self.anchor_query = None
            self.anchor_is_start_location = False
        if self.relation != Relation.NEAR:
            self.max_distance_km = None
        if self.relation != Relation.FAR:
            self.min_distance_km = None
        if self.relation == Relation.NEAR and self.anchor_is_start_location:
            self.anchor_query = None
        return self


class TripSpecification(BaseModel):
    """The structured output of the new intent-extraction step — what the
    Scout now produces instead of choosing tools. See agent/crew.py's
    intent_task for how this is populated via output_pydantic/instructor."""

    parts: list[ItineraryPart] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_sequence_consistency(self):
        indices = [p.sequence_index for p in self.parts]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"sequence_index values must be unique across parts — got {indices}; two parts "
                "cannot both claim the same position in the itinerary"
            )
        return self


def _resolve_anchor(cur, part: ItineraryPart, start_coords: tuple[float, float] | None) -> tuple[str, float, float] | None:
    """Returns (anchor_name, lat, lon) or None. The anchor is the
    traveler's ACTUAL requested reference place — never automatically the
    traveler's own starting location unless anchor_is_start_location was
    explicitly set on this part (see ItineraryPart's own field docstring;
    this is the structural fix for the architecture report's §9 finding
    that a near-relationship's anchor must never silently default to
    start_location)."""
    if part.anchor_is_start_location:
        if start_coords is None:
            return None
        start = get_trip_start()
        return (start.get("label") or "your starting point", start_coords[0], start_coords[1])
    if not part.anchor_query:
        return None
    resolved = _resolve_place(cur, part.anchor_query)
    if resolved is None:
        return None
    _, name, lat, lon = resolved
    return (name, lat, lon)


def _score_one(cur, conn, name: str, part: ItineraryPart, source: str) -> dict | None:
    """The shared scoring step for every relation — calls the EXACT same
    _lookup_place_structured() (real XGBoost inference) every other path
    in this project already uses. Returns a plain dict, not a
    PlaceRecommendation (agent/crew.py owns that schema and does the
    final conversion); None if the resolved name doesn't actually exist
    in the database (defensive — should not normally happen for a row
    that came from _search_places_rows/_places_near/discover_candidates_live)."""
    detail = _lookup_place_structured(cur, conn, name)
    if detail is None:
        return None
    return {
        "name": detail["name"],
        "category": detail["category"],
        "neighborhood": detail["neighborhood"] or "unknown area",
        "opening_hours": detail["opening_hours"],
        "recommendation_confidence": detail["recommendation_confidence"],
        "recommendation_label": detail["recommendation_label"],
        "vibe_cluster": detail["vibe_cluster"],
        "summary": detail["summary"],
        "sources": detail["sources"],
        "distance_km": None, "walk_minutes": None, "bike_minutes": None, "travel_note": None,
        "near_place": None, "near_distance_km": None,
        "sequence_index": part.sequence_index,
        "relation": part.relation.value,
        "source": source,
        "query": part.query,
    }


def _execute_named(cur, conn, part: ItineraryPart, start_coords) -> list[dict]:
    """relation=primary, named_place=true: tries an exact/substring name
    match FIRST (_resolve_place_exact — cheap, precise, and immune to a
    real decoy-name failure mode found live: semantic search alone once
    resolved 'Den lille Havfrue' to a different, similarly-described real
    place). Only when that finds nothing does this fall back to the
    existing relevance-filtered semantic search, and only when THAT also
    finds nothing (internal results insufficient — NAMED_PART_MIN=1) does
    it fall back to live discovery (_live_lookup, the same real
    Nominatim+evidence pipeline search_place_live already uses)."""
    exact = _resolve_place_exact(cur, part.query)
    if exact:
        _, name, lat, lon = exact
        rows = [{"name": name, "lat": lat, "lon": lon}]
        source = "database"
    else:
        rows = _search_places_rows(part.query, category=part.category or "", limit=1)
        source = "database"
        if len(rows) < NAMED_PART_MIN:
            lookup = _live_lookup(part.query)
            if lookup["status"] in ("curated_match", "discovered"):
                rows = _search_places_rows(lookup["name"], limit=1)
                source = "live_discovered" if lookup["status"] == "discovered" else "database"

    results = []
    for row in rows:
        scored = _score_one(cur, conn, row["name"], part, source)
        if scored is None:
            continue
        if start_coords and row.get("lat") is not None and row.get("lon") is not None:
            dist = haversine_km(start_coords[0], start_coords[1], row["lat"], row["lon"])
            scored.update(travel_fields(dist))
        results.append(scored)
    return results


def _execute_open_ended(cur, conn, part: ItineraryPart, start_coords, neighborhood: str) -> list[dict]:
    """relation in (primary-open-ended, sequential, area): plain
    _search_places_rows, with an explicit neighborhood filter for the
    area case (reusing that function's existing, previously-unused-by-
    the-Scout neighborhood parameter — no new SQL). Falls back to
    discover_candidates_live only below OPEN_ENDED_PART_MIN."""
    rows = _search_places_rows(part.query, category=part.category or "", neighborhood=neighborhood, limit=5)
    sources = {r["name"].lower(): "database" for r in rows}

    if len(rows) < OPEN_ENDED_PART_MIN:
        need = OPEN_ENDED_PART_MIN - len(rows) + 1
        extra = discover_candidates_live(part.query, category=part.category or "", neighborhood=neighborhood, limit=need)
        for e in extra:
            key = e["name"].lower()
            if key in sources:
                continue
            rows.append(e)
            sources[key] = "live_discovered"

    results = []
    for row in rows:
        scored = _score_one(cur, conn, row["name"], part, sources.get(row["name"].lower(), "database"))
        if scored is None:
            continue
        if start_coords and row.get("lat") is not None and row.get("lon") is not None:
            dist = haversine_km(start_coords[0], start_coords[1], row["lat"], row["lon"])
            scored.update(travel_fields(dist))
        results.append(scored)
    return results


def _execute_near(cur, conn, part: ItineraryPart, start_coords) -> list[dict]:
    """relation=near: resolves the REAL requested anchor (never
    start_location unless anchor_is_start_location was explicitly set —
    see _resolve_anchor), then calls the existing _places_near() directly
    — the same deterministic haversine ranking search_places_near's @tool
    already used, just invoked by the backend instead of an LLM tool
    call. If the anchor itself can't be resolved at all, degrades to a
    plain open-ended search rather than silently inventing a location,
    and flags that degradation honestly via execution_note."""
    anchor = _resolve_anchor(cur, part, start_coords)
    if anchor is None:
        log.info(f"{part.query!r}: relation=near but its anchor could not be resolved — falling back to a plain open-ended search")
        results = _execute_open_ended(cur, conn, part, start_coords, neighborhood="")
        for r in results:
            r["execution_note"] = "requested near a reference place that could not be resolved"
        return results

    anchor_name, anchor_lat, anchor_lon = anchor
    rows = _places_near(
        cur, anchor_lat, anchor_lon, category=part.category or "",
        exclude_name=anchor_name, limit=3, max_km=part.max_distance_km,
    )
    sources = {r["name"].lower(): "database" for r in rows}

    if len(rows) < NEAR_PART_MIN:
        need = NEAR_PART_MIN - len(rows) + 1
        extra = discover_candidates_live(part.query, category=part.category or "", limit=need)
        radius = part.max_distance_km if part.max_distance_km is not None else MAX_NEARBY_KM
        for e in extra:
            key = e["name"].lower()
            if key in sources:
                continue
            dist = haversine_km(anchor_lat, anchor_lon, e["lat"], e["lon"])
            # Never blindly trust a Serper candidate for a near
            # relationship — it must genuinely satisfy the traveler's own
            # distance limit (or the default radius), checked
            # independently here, not assumed from Serper's own ranking.
            if dist > radius:
                continue
            e = dict(e)
            e["distance_km"] = dist
            rows.append(e)
            sources[key] = "live_discovered"

    results = []
    for row in rows:
        scored = _score_one(cur, conn, row["name"], part, sources.get(row["name"].lower(), "database"))
        if scored is None:
            continue
        scored["near_place"] = anchor_name
        scored["near_distance_km"] = round(row["distance_km"], 2)
        results.append(scored)
    return results


def _execute_far(cur, conn, part: ItineraryPart, start_coords) -> list[dict]:
    """relation=far: deliberately NEVER calls _places_near — uses the
    plain, city-wide _search_places_rows exactly like an open-ended part
    with no reference point, which is what 'far' already meant in the
    prior, verified-working prompt-only behavior (PROJECT_ARCHITECTURE_
    REPORT.md §12 item 4). If the traveler gave an explicit numeric
    min_distance_km, that's the ONLY numeric 'far' constraint this
    project invents — a qualitative 'far away' with no number is honored
    simply by never anchoring the search, not by guessing a default
    distance threshold."""
    anchor = _resolve_anchor(cur, part, start_coords)  # informational only — NEVER restricts the search
    rows = _search_places_rows(part.query, category=part.category or "", limit=5)
    sources = {r["name"].lower(): "database" for r in rows}

    if len(rows) < OPEN_ENDED_PART_MIN:
        need = OPEN_ENDED_PART_MIN - len(rows) + 1
        extra = discover_candidates_live(part.query, category=part.category or "", limit=need)
        for e in extra:
            key = e["name"].lower()
            if key in sources:
                continue
            rows.append(e)
            sources[key] = "live_discovered"

    results = []
    for row in rows:
        lat, lon = row.get("lat"), row.get("lon")
        if anchor is not None and part.min_distance_km is not None and lat is not None and lon is not None:
            _anchor_name, anchor_lat, anchor_lon = anchor
            if haversine_km(anchor_lat, anchor_lon, lat, lon) < part.min_distance_km:
                continue
        scored = _score_one(cur, conn, row["name"], part, sources.get(row["name"].lower(), "database"))
        if scored is None:
            continue
        if start_coords and lat is not None and lon is not None:
            dist = haversine_km(start_coords[0], start_coords[1], lat, lon)
            scored.update(travel_fields(dist))
        results.append(scored)
    return results


def _execute_part(cur, conn, part: ItineraryPart, start_coords) -> list[dict]:
    if part.relation == Relation.NEAR:
        return _execute_near(cur, conn, part, start_coords)
    if part.relation == Relation.FAR:
        return _execute_far(cur, conn, part, start_coords)
    if part.relation == Relation.AREA:
        return _execute_open_ended(cur, conn, part, start_coords, neighborhood=part.neighborhood or "")
    if part.relation == Relation.SEQUENTIAL:
        return _execute_open_ended(cur, conn, part, start_coords, neighborhood="")
    # PRIMARY
    if part.named_place:
        return _execute_named(cur, conn, part, start_coords)
    return _execute_open_ended(cur, conn, part, start_coords, neighborhood="")


def execute_trip_specification(spec: TripSpecification) -> list[dict]:
    """The deterministic backend execution layer — for each ItineraryPart,
    in the traveler's requested order, calls the EXISTING search function
    the part's relation implies (never an LLM decision), scores every
    candidate through the real XGBoost pipeline, and computes real
    haversine distances. No LLM call anywhere in this function.

    Reads the trip's start coordinates via get_trip_start() (set once by
    agent/crew.py's plan_trip(), same as every other tool in this
    project) rather than taking them as a parameter — matches the
    existing module-level pattern travel_time_estimate/_ensure_start_
    distance already use.

    Returns a flat list of plain dicts, in the same order as
    spec.parts sorted by sequence_index — agent/crew.py converts these
    into PlaceRecommendation objects; this module never constructs that
    schema itself (see this module's own docstring for why)."""
    start = get_trip_start()
    start_coords = (start["lat"], start["lon"]) if start.get("lat") is not None else None

    results: list[dict] = []
    with connect() as conn, conn.cursor() as cur:
        for part in sorted(spec.parts, key=lambda p: p.sequence_index):
            results.extend(_execute_part(cur, conn, part, start_coords))
    return results
