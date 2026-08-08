"""Phase 11 — tools wrapping data from earlier phases: pgvector place search
(6), quality-score predictions (9), RAG summaries (8), and weather +
outdoor-interest forecasts (10). Covers the tool set docs/architecture.md
describes for the trip-planning crew: SQL, pgvector, weather, quality-score.

Built with crewai.tools.tool, not a LangChain tool wrapper: the original
Phase 0 plan assumed CrewAI wrapped LangChain tools directly, but crewai
1.15.8's Agent.tools is typed to crewai.tools.base_tool.BaseTool specifically
(confirmed by inspecting the installed package, not assumed) — a LangChain
BaseTool doesn't satisfy that type. LangChain's real, verified use in this
project remains Phase 8's RAG chain; see docs/architecture.md for the
corrected account of Phase 11.

Each tool opens its own short-lived DB connection, matching every other
script in this project — fine at this project's demo scale (a handful of
tool calls per trip-plan request), not meant to survive real production
request volume.

Query embedding uses fastembed (ONNX Runtime), not sentence-transformers —
crewai itself doesn't need torch (confirmed: `import crewai` alone never
touches sys.modules for torch/transformers), but sentence-transformers does,
and torch pushed this service's memory past Render's free-tier 512MB limit
(deploy was getting OOM-killed, exit 137). fastembed's ONNX export of the
same all-MiniLM-L6-v2 weights produces effectively identical vectors
(cosine similarity 1.0000 against sentence-transformers' output, verified
before switching) — same embedding space as what Phase 6's batch ingestion
script already wrote to the DB, just a lighter runtime for the live API.
That batch script keeps using sentence-transformers; it runs locally, not
on Render, so its memory footprint was never the problem.

connect() and get_embed_model() are also imported directly by api/main.py's
/explore and /stats routes — render.yaml runs one process, and api/main.py
already imports agent.crew, which already imports this module, so that
process already holds one fastembed instance. Sharing these two helpers
keeps it at one instance/one connection pattern instead of a second,
memory-doubling copy in a second "private" module.
"""

import functools
import logging
import math
import os
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg
import requests
from crewai.tools import tool
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

load_dotenv()

log = logging.getLogger("tools")

# Set once per plan_trip() call, before crew.kickoff() — a module-level
# global, not a per-request object, because crewai's @tool functions are
# plain module-level functions with no natural place to inject per-request
# state. Same "fine at this project's demo scale" trade-off as the DB
# connection pattern above: a real concurrent-request server would need
# this threaded through properly instead. See agent/crew.py's plan_trip().
_trip_start: dict = {"lat": None, "lon": None, "label": None}


def set_trip_start(lat: float | None, lon: float | None, label: str) -> None:
    _trip_start["lat"], _trip_start["lon"], _trip_start["label"] = lat, lon, label


# Per-request memoization for tool calls — a real, deterministic backstop
# against wasted repeated work within one plan_trip() run, independent of
# whether an agent's own reasoning ever decides to stop asking for the
# same thing twice. Reset at the start of every plan_trip() call (see
# agent/crew.py), not persisted across requests — this is about one
# run's redundant calls, not a cross-request cache (weather_daily/
# reviews_raw already handle real cross-request caching where it matters).
_tool_call_cache: dict[tuple, str] = {}


def reset_tool_call_cache() -> None:
    _tool_call_cache.clear()


def get_cached_tool_calls(fn_name: str) -> list[dict]:
    """The kwargs of every call to the named tool that this request has
    already made and cached (see _tool_call_cache above) — used by
    agent/crew.py's Groq-failure cache-reuse path to recover which places a
    partially-completed crew run had already resolved, without that module
    reaching into _tool_call_cache's internal (fn_name, sorted kwargs)
    tuple-key structure directly."""
    return [dict(key[1]) for key in _tool_call_cache if key[0] == fn_name]


def _cache_tool_calls(fn):
    """Wraps a tool's real implementation (applied BELOW @tool, so @tool
    still introspects the original signature/docstring — verified this
    survives functools.wraps correctly) — identical (tool, kwargs) calls
    within the same request return the cached result instead of repeating
    a real DB query or live web fetch."""
    @functools.wraps(fn)
    def wrapper(**kwargs):
        key = (fn.__name__, tuple(sorted(kwargs.items())))
        if key in _tool_call_cache:
            log.info(f"{fn.__name__}({kwargs}): identical call already made this request, reusing result")
            return _tool_call_cache[key]
        result = fn(**kwargs)
        _tool_call_cache[key] = result
        return result
    return wrapper


# search_places_near always returned its closest N candidates regardless of
# how far away "closest" actually was — fine when the nearest match is
# genuinely close, but in a sparse area the "closest" cafe could still be
# several km off, and returning it unqualified would call something "near"
# that no traveler would call walkable. Capping the radius means the tool
# can honestly say "nothing genuinely nearby" instead of quietly relabeling
# "not that close" as "near."
MAX_NEARBY_KM = 2.0


# Average speeds used for the estimate, not a routed distance — deliberately
# not a real transit API. Checked Rejseplanen (Denmark's official journey
# planner, the "right" real data source here): it requires a manual
# approval process (contact form, wait for a human, then set a password),
# not an instant free key, so it's a documented future upgrade, not
# something to block this feature on. Copenhagen is flat and very
# bike-friendly, so these are reasonable real-world averages, not
# arbitrary guesses.
WALK_KMH = 5.0
BIKE_KMH = 15.0

# Past this many minutes, walking stops being a realistic suggestion —
# nudge toward public transit instead, honestly (no fake bus/line number,
# since that needs real routing data this project doesn't have access to).
WALK_MINUTES_LIMIT = 15


def travel_fields(dist_km: float) -> dict:
    """The actual walk/bike/note computation, shared by the live
    travel_time_estimate tool below and agent/crew.py's trip-plan cache
    (which recomputes these for a new starting point without re-running the
    crew — pure math, zero LLM cost). Kept in one place so both stay
    consistent instead of two copies drifting apart over time."""
    walk_min = round(dist_km / WALK_KMH * 60)
    bike_min = round(dist_km / BIKE_KMH * 60)
    travel_note = (
        "too far to walk comfortably, consider biking or transit"
        if walk_min > WALK_MINUTES_LIMIT
        else None
    )
    return {
        "distance_km": round(dist_km, 2),
        "walk_minutes": walk_min,
        "bike_minutes": bike_min,
        "travel_note": travel_note,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model: TextEmbedding | None = None


def get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model


def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=15)
    register_vector(conn)
    return conn


def _coerce_limit(limit: int | str, default: int = 5) -> int:
    """search_places/top_quality_places declare limit as int | str, not just
    int — found live: Groq/Llama's tool-call generation sometimes sends
    numeric-looking arguments as JSON strings (e.g. "5" instead of 5), and
    Groq's own server-side schema validation rejects the whole call outright
    when that doesn't match a strict `integer` type, before our code ever
    runs (confirmed: the rejection is a GroqException from the API itself,
    not a local error we could catch and coerce after the fact). Widening
    the declared type to accept what the model actually tends to send, and
    coercing here, fixes this at the only point it can be fixed."""
    try:
        return int(limit)
    except (TypeError, ValueError):
        return default


def _resolve_place(cur, place_name: str) -> tuple | None:
    """Returns (place_id, name, lat, lon) for the best-matching place, or
    None. Tries an exact/substring name match first (cheap, precise); falls
    back to semantic search when that finds nothing.

    The fallback is the real fix for a bug found live: the Concierge asked
    place_details for "Little Mermaid statue" (the traveler's English
    wording), but the DB stores this landmark under its Danish OSM name,
    "Den lille Havfrue" — zero literal text overlap, so ILIKE found nothing
    even though the place and its full AI summary genuinely exist. Semantic
    search doesn't care what language/wording was used, only meaning."""
    cur.execute(
        "SELECT place_id, name, lat, lon FROM places WHERE name ILIKE %(name)s "
        "ORDER BY (lower(name) = lower(%(exact)s)) DESC LIMIT 1;",
        {"name": f"%{place_name}%", "exact": place_name},
    )
    row = cur.fetchone()
    if row:
        return row

    model = get_embed_model()
    query_embedding = next(model.embed([place_name]))
    cur.execute(
        "SELECT place_id, name, lat, lon FROM places WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %(qvec)s LIMIT 1;",
        {"qvec": query_embedding},
    )
    return cur.fetchone()


def _search_places_rows(query: str, category: str = "", neighborhood: str = "", limit: int | str = 5) -> list[dict]:
    """The real search_places query, factored out so agent/crew.py's
    deterministic_trip_plan() (used when Groq is confirmed unavailable —
    see that function's docstring) can get real candidate places without
    going through the @tool/LLM-calling layer at all. search_places()
    below is now a thin text-formatting wrapper around this.

    Real gap this used to have, same class as the one already fixed for
    /explore: raw "ORDER BY embedding <=> qvec LIMIT n" treats every
    result in the requested limit as equally relevant. Pulls a wider pool
    than what's requested and reranks with api.ranking's deterministic
    layer (real lexical name matching + category-intent detection, not
    semantic similarity alone) before capping to the caller's limit — so
    BOTH the normal Scout-driven path (search_places) and the Groq-
    unavailable deterministic fallback get the same relevance quality,
    not just Explore."""
    from api.ranking import DEFAULT_POOL_SIZE, rank_explore_candidates

    model = get_embed_model()
    query_embedding = next(model.embed([query]))
    sql = """
        SELECT name, category, subcategory, neighborhood, opening_hours, lat, lon,
               1 - (embedding <=> %(qvec)s) AS similarity
        FROM places
        WHERE embedding IS NOT NULL
    """
    params = {"qvec": query_embedding, "limit": DEFAULT_POOL_SIZE}
    if category:
        sql += " AND category = %(category)s"
        params["category"] = category
    if neighborhood:
        sql += " AND neighborhood = %(neighborhood)s"
        params["neighborhood"] = neighborhood
    sql += " ORDER BY embedding <=> %(qvec)s LIMIT %(limit)s;"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    return rank_explore_candidates(query, rows, final_limit=_coerce_limit(limit))


@tool
@_cache_tool_calls
def search_places(query: str, category: str = "", neighborhood: str = "", limit: int | str = 5) -> str:
    """Semantic search over Copenhagen places (pgvector). Use this to find
    candidate places matching a vibe or description, e.g. "cozy quiet cafe
    good for working". Optionally filter by category (restaurant, cafe,
    hotel, landmark, bar) and/or neighborhood."""
    rows = _search_places_rows(query, category, neighborhood, limit)
    if not rows:
        return "No matching places found."
    return "\n".join(
        f"- {r['name']} ({r['category']}, {r['neighborhood'] or 'unknown area'}), "
        f"similarity {r['similarity']:.2f}, hours: {r['opening_hours'] or 'unknown'}"
        for r in rows
    )


def _places_near(cur, lat: float, lon: float, category: str = "", exclude_name: str | None = None, limit: int = 3) -> list[dict]:
    """Real places within MAX_NEARBY_KM of (lat, lon), sorted by actual
    haversine distance — the shared core of search_places_near() below,
    factored out so agent/crew.py's deterministic_trip_plan() can rank
    "near X" candidates by real geographic distance FROM A NAMED PLACE
    (never from the traveler's own start_location, and never by semantic
    similarity) without going through the @tool/LLM-calling layer."""
    sql = "SELECT name, category, neighborhood, opening_hours, lat, lon FROM places WHERE lat IS NOT NULL AND lon IS NOT NULL"
    params: dict = {}
    if exclude_name:
        sql += " AND lower(name) != lower(%(exclude_name)s)"
        params["exclude_name"] = exclude_name
    if category:
        sql += " AND category = %(category)s"
        params["category"] = category
    cur.execute(sql, params)
    columns = [d.name for d in cur.description]
    rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    for r in rows:
        r["distance_km"] = haversine_km(lat, lon, r["lat"], r["lon"])
    rows.sort(key=lambda r: r["distance_km"])
    return [r for r in rows if r["distance_km"] <= MAX_NEARBY_KM][:limit]


@tool
@_cache_tool_calls
def search_places_near(anchor_place: str, category: str = "", limit: int | str = 3) -> str:
    """Finds real places near ANOTHER named place, ranked by actual
    geographic distance — not text/semantic similarity, which only
    approximates proximity through wording and can return places that
    aren't really close. Use this whenever the request says something is
    near/close to/around another specific named place (e.g. "coffee near
    the Little Mermaid", "a hotel near Torvehallerne") — use search_places
    instead for a request with no such reference point. Optionally filter
    by category (restaurant, cafe, hotel, landmark, bar)."""
    limit = _coerce_limit(limit)
    with connect() as conn, conn.cursor() as cur:
        anchor = _resolve_place(cur, anchor_place)
        if not anchor:
            return f"No place found matching '{anchor_place}' to search near."
        anchor_id, anchor_name, anchor_lat, anchor_lon = anchor
        rows = _places_near(cur, anchor_lat, anchor_lon, category=category, exclude_name=anchor_name, limit=limit)

    if not rows:
        scope = f" in category '{category}'" if category else ""
        return (
            f"No places found within {MAX_NEARBY_KM:.0f} km of {anchor_name}{scope} — "
            "nothing genuinely nearby, say so honestly rather than suggesting somewhere far away."
        )
    return "\n".join(
        f"- {r['name']} ({r['category']}, {r['neighborhood'] or 'unknown area'}), "
        f"{r['distance_km']:.2f} km from {anchor_name}, hours: {r['opening_hours'] or 'unknown'}"
        for r in rows
    )


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "ai-denmark-explorer/0.1 (Copenhagen pilot, personal project)"}

# Same 1 request/second policy as ingestion/osm_live_lookup.py and this
# module's own trip-start geocoding — a module-level timestamp is enough
# since a single trip-plan request only ever calls this a handful of times.
_last_live_lookup = 0.0


def _respect_nominatim_rate_limit() -> None:
    global _last_live_lookup
    elapsed = time.time() - _last_live_lookup
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    _last_live_lookup = time.time()


# A real place Nominatim resolves might already be a curated place under
# different wording ("The Little Mermaid" vs the DB's own "Den lille
# Havfrue") — verified live: Nominatim's real osm_id for that landmark
# (node/25074274) matches the curated row's osm_id exactly, same
# coordinates. Checked first, since it's a perfect identity match when
# available; ~50m is generous enough to absorb real GPS/geocoding jitter
# for results Nominatim resolves without a clean OSM id, tight enough not
# to match a genuinely different nearby place.
_COORD_MATCH_RADIUS_KM = 0.05


def _find_curated_match(cur, osm_type: str | None, osm_id: int | None, lat: float, lon: float) -> tuple | None:
    """Returns (place_id, name) if a live Nominatim result is actually
    already a curated place, else None."""
    if osm_type and osm_id:
        cur.execute("SELECT place_id, name FROM places WHERE osm_id = %s LIMIT 1;", (f"{osm_type}/{osm_id}",))
        row = cur.fetchone()
        if row:
            return row

    lat_delta = _COORD_MATCH_RADIUS_KM / 111.0  # ~km per degree of latitude
    lon_delta = _COORD_MATCH_RADIUS_KM / (111.0 * math.cos(math.radians(lat)) or 1.0)
    cur.execute(
        "SELECT place_id, name, lat, lon FROM places "
        "WHERE lat BETWEEN %s AND %s AND lon BETWEEN %s AND %s;",
        (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta),
    )
    for place_id, name, plat, plon in cur.fetchall():
        if haversine_km(lat, lon, plat, plon) <= _COORD_MATCH_RADIUS_KM:
            return (place_id, name)
    return None


# Best-effort, deliberately conservative mapping from Nominatim's real OSM
# category/type taxonomy to this project's five-category taxonomy — real,
# not guessed, but genuinely imperfect (OSM's tagging vocabulary is much
# richer than five buckets), which is exactly why a live-discovered place
# is marked data_status='live_discovered' rather than blended silently
# into curated data. A (category, type) pair not listed here returns None
# — callers must treat that as "can't safely persist this," not fall back
# to a guessed default category just to make a row insertable, since the
# category one-hot is a real model feature, not cosmetic.
_NOMINATIM_CATEGORY_MAP = {
    ("amenity", "restaurant"): "restaurant", ("amenity", "fast_food"): "restaurant",
    ("amenity", "cafe"): "cafe",
    ("amenity", "bar"): "bar", ("amenity", "pub"): "bar", ("amenity", "biergarten"): "bar",
    ("tourism", "hotel"): "hotel", ("tourism", "guest_house"): "hotel", ("tourism", "hostel"): "hotel",
    ("tourism", "attraction"): "landmark", ("tourism", "artwork"): "landmark",
    ("tourism", "museum"): "landmark", ("tourism", "gallery"): "landmark",
}
_NOMINATIM_CATEGORY_CLASS_FALLBACK = {"historic": "landmark", "leisure": "landmark"}


def _map_nominatim_category(category: str, place_type: str) -> str | None:
    mapped = _NOMINATIM_CATEGORY_MAP.get((category, place_type))
    if mapped:
        return mapped
    return _NOMINATIM_CATEGORY_CLASS_FALLBACK.get(category)


def _discover_live_place(conn, resolved_name: str, mapped_category: str, lat: float, lon: float, osm_type: str | None, osm_id: int | None) -> tuple[str, list[str]] | None:
    """A place Nominatim resolved that is genuinely new — no curated match
    even by coordinates. Real evidence is gathered FIRST, with no DB write
    at all; a new places row is only ever inserted once that evidence
    search actually found something, matching the explicit requirement not
    to blindly persist every Nominatim hit. Returns (place_id, review
    texts) on success, or None if no usable evidence was found — callers
    must report that honestly as insufficient evidence, not silently
    proceed to a zero-evidence prediction."""
    evidence = _search_place_evidence(resolved_name, official_website=None)
    if not evidence:
        return None

    osm_id_str = f"{osm_type}/{osm_id}" if osm_type and osm_id else f"live/{uuid.uuid4()}"
    model = get_embed_model()
    embedding = next(model.embed([resolved_name]))

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO places (osm_id, name, category, lat, lon, embedding, data_status, source_url)
            VALUES (%s, %s, %s, %s, %s, %s, 'live_discovered', %s)
            RETURNING place_id;
            """,
            (osm_id_str, resolved_name, mapped_category, lat, lon, embedding, evidence[0]["link"]),
        )
        place_id = cur.fetchone()[0]
        found_texts = _store_place_evidence(cur, place_id, resolved_name, evidence)
    conn.commit()
    log.info(f"{resolved_name!r}: discovered live, persisted as a new place ({len(found_texts)} evidence source(s))")
    return str(place_id), found_texts


@tool
@_cache_tool_calls
def search_place_live(query: str) -> str:
    """Last-resort live lookup for a named place — use ONLY after
    search_places, search_places_near, and top_quality_places all found
    nothing. Checks whether the resolved place is actually already
    curated under different wording (use place_details for it as normal
    if so); otherwise attempts to discover and score it from real web
    evidence. If that succeeds, it now has a real recommendation
    confidence — call place_details for it like any other place. If
    evidence genuinely can't be found, says so honestly; never presents an
    invented or default confidence."""
    _respect_nominatim_rate_limit()
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": f"{query}, Copenhagen, Denmark", "format": "jsonv2", "limit": 1},
            headers=NOMINATIM_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        return f"Live lookup failed for '{query}': {e}"

    if not results:
        return f"No live result found for '{query}' either — it may not exist in Copenhagen."

    r = results[0]
    name = r.get("name") or query
    address = r.get("display_name", "address unknown")
    lat, lon = float(r["lat"]), float(r["lon"])
    osm_type, osm_id = r.get("osm_type"), r.get("osm_id")

    with connect() as conn, conn.cursor() as cur:
        curated = _find_curated_match(cur, osm_type, osm_id, lat, lon)
        if curated:
            _, curated_name = curated
            return (
                f"'{query}' is already in our curated database as '{curated_name}' — "
                f"call place_details for '{curated_name}' to get its real recommendation "
                "confidence, vibe cluster, and AI summary; do not treat this as a live-only result."
            )

        mapped_category = _map_nominatim_category(r.get("category", ""), r.get("type", ""))
        if not mapped_category:
            return (
                f"LIVE LOOKUP RESULT (not in our curated dataset, category unclear — no "
                f"recommendation confidence, vibe cluster, or AI summary available): {name}, {address}."
            )

        discovered = _discover_live_place(conn, name, mapped_category, lat, lon, osm_type, osm_id)

    if discovered is None:
        return (
            f"LIVE LOOKUP RESULT (not in our curated dataset, and no real review/evidence "
            f"text could be found for it — no recommendation confidence available; treat as "
            f"unscored, do not invent a confidence): {name}, {address}."
        )

    _, found_texts = discovered
    return (
        f"'{query}' was found and added to our database as '{name}' — real evidence was found "
        f"({len(found_texts)} source(s)) and a real recommendation confidence is now available. "
        f"Call place_details for '{name}' to get it."
    )


@tool
@_cache_tool_calls
def top_quality_places(category: str = "", neighborhood: str = "", limit: int | str = 5) -> str:
    """Ranks Copenhagen places by predicted quality score (0-100, Phase 9
    XGBoost model), optionally filtered by category and/or neighborhood. Use
    this when the request is about the *best-rated* places rather than a
    specific vibe."""
    sql = """
        SELECT p.name, p.category, p.neighborhood, m.predicted_value AS quality_score
        FROM places p
        JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
        WHERE 1=1
    """
    params: dict = {"limit": _coerce_limit(limit)}
    if category:
        sql += " AND p.category = %(category)s"
        params["category"] = category
    if neighborhood:
        sql += " AND p.neighborhood = %(neighborhood)s"
        params["neighborhood"] = neighborhood
    sql += " ORDER BY m.predicted_value DESC LIMIT %(limit)s;"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    if not rows:
        return "No places matched those filters."
    return "\n".join(
        f"- {r['name']} ({r['category']}, {r['neighborhood'] or 'unknown area'}), "
        f"quality score {r['quality_score']:.1f}/100"
        for r in rows
    )


_WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKIPEDIA_HEADERS = {"User-Agent": "ai-denmark-explorer/0.1 (Copenhagen pilot, personal project)"}
_MIN_FALLBACK_TEXT_CHARS = 40


def _mentions_copenhagen_or_denmark(text: str) -> bool:
    """The relevance gate every live-fetched fallback text must pass before
    being stored. Real problem this caught, not hypothetical: a live test
    of a generically-named landmark ('Abstrakt skulptur' — Danish for
    'abstract sculpture', not a unique proper noun) had its Serper results
    confidently link three e-commerce listings for decorative sculpture
    products — completely unrelated to the real Copenhagen public artwork —
    because they only passed length/domain filtering, nothing about actual
    topical relevance. Requiring an explicit Copenhagen/Denmark mention in
    the text itself is a real, if imperfect, check: it won't catch every
    false positive, but it directly would have rejected all three bad
    results above. Applied identically to Wikipedia and Serper results so
    neither source gets a looser bar than the other."""
    lowered = text.lower()
    return "copenhagen" in lowered or "denmark" in lowered or "københavn" in lowered


def _wikipedia_summary(place_name: str) -> dict | None:
    """Live, keyless, free Wikipedia REST API lookup for one place name —
    used as the last-resort tier in _fetch_place_knowledge below, not the
    first. Wikipedia is stable and factual, but real evidence from this
    project's own testing shows it's a poor fit for a system whose job is
    answering "why should I visit this place," not "what is this place":
    a live test against 'Absalon' (a real Copenhagen landmark honoring the
    historical bishop) correctly found a real, on-topic Wikipedia summary —
    but it's a biography, not a description of the landmark or why a
    traveler would visit, and the recommendation classifier scored it only
    49%/"not recommended" partly as a result. Kept as a real fallback for
    when nothing better exists, not the default answer.

    Deliberately NOT Wikivoyage here — ingestion/wikivoyage_descriptions.py
    already ran a full batch pass linking Wikivoyage's structured Copenhagen
    listings against every curated place; a place reaching this live
    fallback has already been checked against Wikivoyage and found nothing
    there, so re-trying it live would just repeat a known-failed search.

    Returns None (not an exception) for anything not confidently on-topic:
    no page, a disambiguation page, or a summary that never even mentions
    Copenhagen/Denmark."""
    try:
        resp = requests.get(
            _WIKIPEDIA_SUMMARY_URL.format(title=place_name.replace(" ", "_")),
            headers=_WIKIPEDIA_HEADERS,
            timeout=10,
        )
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if data.get("type") == "disambiguation":
        return None
    extract = data.get("extract", "")
    if len(extract) < _MIN_FALLBACK_TEXT_CHARS:
        return None
    if not _mentions_copenhagen_or_denmark(extract):
        return None
    return {
        "title": data.get("title", place_name),
        "text": extract,
        "source_url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page")
        or f"https://en.wikipedia.org/wiki/{place_name.replace(' ', '_')}",
        "tier": "wikipedia",
    }


# Real Copenhagen tourism-organization domains — traveler-recommendation
# focused ("why visit"), not encyclopedic ("what is it"), the actual gap
# Wikipedia-only fallback left. Deliberately a short, hand-verified
# allowlist rather than a heuristic guess at "looks official" — a wrong
# match here would be worse than finding nothing.
_TRUSTED_TOURISM_DOMAINS = ("visitcopenhagen.com", "wonderfulcopenhagen.dk")


def _domain_tier(url: str, official_domain: str | None) -> int:
    """Ranks a search result by source quality, lower is better. 0 = the
    place's own official website (from OSM's real `website` tag on this
    exact place — the strongest possible signal, not a guess); 1 = a known
    Copenhagen tourism organization (traveler-focused by design); 2 =
    everything else that already passed the relevance filter."""
    if official_domain and official_domain in url:
        return 0
    if any(d in url for d in _TRUSTED_TOURISM_DOMAINS):
        return 1
    return 2


def _search_place_evidence(place_name: str, official_website: str | None) -> list[dict]:
    """The real, place-specific evidence search — site-scoped official
    search first, then general Copenhagen-scoped search ranked by source
    quality (known tourism organizations above generic snippets), then
    Wikipedia only if nothing above found anything usable. Pure: no DB
    access, no side effects, so both the curated-place enrichment path
    (_fetch_place_knowledge, below) and agent/crew.py's live-discovery
    path (_discover_live_place) share this exact search behavior instead
    of two copies that could drift apart. Returns a list of dicts with at
    least title/link/snippet/tier/source_type — empty if nothing was found
    or every provider call failed; callers are responsible for treating
    that honestly (no evidence found is a real, valid outcome, not an
    error to retry).

    Real, live-tested reason the official-site check is a SEPARATE,
    site-scoped search rather than just hoping it ranks well in general
    results: tested against a real database record — a 1901 statue of
    Bishop Absalon, whose OSM data already names its real official source
    (samlingen.koes.dk, Copenhagen's public art registry) — a plain
    "Absalon Copenhagen" search never surfaced that niche page at all,
    instead ranking a popular, unrelated, same-named venue (a converted-
    church community space) highly enough to look like a confident tourism-
    org match. A `site:`-scoped search is what actually finds it."""
    from ingestion.web_enrichment import filter_results, search_web

    # search_web() reads os.environ["SERPER_API_KEY"] directly and raises
    # a bare KeyError (not requests.exceptions.RequestException) when it's
    # unset — that KeyError was propagating uncaught through this whole
    # call chain (live discovery / place_details) and surfacing as an
    # HTTP 500 on /trip-plan, which the frontend only shows as "Failed to
    # fetch". Missing key is a real, expected local/deploy state, not an
    # error — skip straight to the existing keyless Wikipedia fallback.
    has_serper_key = bool(os.environ.get("SERPER_API_KEY"))

    official_domain = None
    if official_website:
        official_domain = official_website.split("//")[-1].split("/")[0].removeprefix("www.")

    results = []
    if official_domain and has_serper_key:
        try:
            results = filter_results(search_web(f"site:{official_domain} {place_name}"))
        except requests.exceptions.RequestException as e:
            log.info(f"Live official-site search failed for {place_name!r}: {e}")
        # A site:-scoped result is already confirmed on the place's own
        # official domain — the general Copenhagen/Denmark relevance gate
        # below would still reject a page that never happens to say either
        # word (a real risk for a niche registry entry), so it's checked
        # separately here rather than folded into the shared filter.
        results = [r for r in results if official_domain in r["link"]]

    if not results and has_serper_key:
        try:
            general_results = filter_results(search_web(f"{place_name} Copenhagen"))
        except requests.exceptions.RequestException as e:
            log.info(f"Live Serper search failed for {place_name!r}: {e}")
            general_results = []
        # web_enrichment.filter_results only screens length/low-signal
        # domains, not actual topical relevance — real, observed failure
        # mode this closes: a generic place name (no unique proper noun)
        # matching unrelated commercial results that just share descriptive
        # words. See _mentions_copenhagen_or_denmark's docstring for the
        # concrete case. Not applied to the official-site results above —
        # a site:-scoped match is already confirmed on-domain, and a niche
        # registry page (the real Absalon case) may never say "Copenhagen"
        # explicitly even when it's exactly the right page.
        results = [r for r in general_results if _mentions_copenhagen_or_denmark(r.get("snippet", ""))]
    elif not has_serper_key:
        log.info(f"SERPER_API_KEY not configured — skipping live web search for {place_name!r}")

    tier_names = {0: "official_site", 1: "tourism_org", 2: "search_snippet"}
    evidence = [
        {**r, "source_type": "web_search", "tier": tier_names[_domain_tier(r["link"], official_domain)]}
        for r in sorted(results, key=lambda r: _domain_tier(r["link"], official_domain))
    ]

    if not evidence:
        wiki = _wikipedia_summary(place_name)
        if wiki:
            evidence = [{
                "title": wiki["title"], "link": wiki["source_url"], "snippet": wiki["text"],
                "source_type": "wikipedia", "tier": "wikipedia",
            }]
    return evidence


def _store_place_evidence(cur, place_id, place_name: str, evidence: list[dict]) -> list[str]:
    """Writes already-fetched evidence (from _search_place_evidence) as
    real reviews_raw rows tied to place_id. Split out from
    _fetch_place_knowledge so a caller that already has evidence in hand
    (the live-discovery path below) doesn't need to search Serper twice
    for the same place."""
    run_id = f"live-fallback-{datetime.now().strftime('%Y%m%d-%H%M%S')}"  # noqa: DTZ005
    found_texts: list[str] = []
    for e in evidence:
        cur.execute(
            """
            INSERT INTO reviews_raw (place_id, source_type, source_id, source_url,
                                      text_content, run_id, raw_payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING review_id;
            """,
            (place_id, e["source_type"], e.get("title", place_name), e["link"], e["snippet"], run_id, psycopg.types.json.Json(e)),
        )
        review_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO place_mentions (review_id, place_id, match_method, confidence) "
            "VALUES (%s, %s, 'manual', 1.0);",
            (review_id, place_id),
        )
        found_texts.append(e["snippet"])
    return found_texts


def _fetch_place_knowledge(conn, place_id, place_name: str, official_website: str | None) -> list[str]:
    """Last-resort live enrichment — called ONLY when a curated place has
    zero linked reviews_raw text (checked by the caller before this runs).
    Whatever _search_place_evidence finds is stored tagged with which real
    tier it came from in raw_payload — so this place never needs live
    enrichment again, and it's always possible to tell later whether a
    description came from the official site, a tourism org, or a generic
    snippet/Wikipedia. Returns the newly found text so the CURRENT request
    can use it immediately, not just the next one."""
    evidence = _search_place_evidence(place_name, official_website)
    with conn.cursor() as cur:
        found_texts = _store_place_evidence(cur, place_id, place_name, evidence)
    conn.commit()
    if found_texts:
        log.info(f"{place_name!r}: live fallback found {len(found_texts)} new source(s)")
    return found_texts


def _lookup_place_structured(cur, conn, place_name: str) -> dict | None:
    """The real place_details lookup for ONE place, factored out so
    agent/crew.py's deterministic_trip_plan() (used when Groq is
    confirmed unavailable) can get real recommendation data — including a
    live predict_recommendation() call, which needs no Groq/LLM at all —
    without going through the @tool/LLM-calling layer. place_details()
    below is now a thin per-place text-formatting wrapper around this."""
    resolved = _resolve_place(cur, place_name)
    if not resolved:
        return None
    place_id = resolved[0]

    cur.execute(
        """
        SELECT p.place_id, p.name, p.category, p.subcategory, p.neighborhood,
               p.opening_hours, p.price_level, p.osm_tags,
               m.predicted_value AS old_quality_score,
               d.predicted_value AS distilbert_sentiment_score,
               c.label AS cluster_label
        FROM places p
        LEFT JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
        LEFT JOIN ml_predictions d ON d.place_id = p.place_id AND d.target = 'distilbert_sentiment'
        LEFT JOIN place_clusters pc ON pc.place_id = p.place_id
        LEFT JOIN clusters c ON c.cluster_id = pc.cluster_id
        WHERE p.place_id = %s
        LIMIT 1;
        """,
        (place_id,),
    )
    row = cur.fetchone()
    columns = [d.name for d in cur.description]
    place = dict(zip(columns, row))

    cur.execute(
        "SELECT aspect_category, avg_score, num_mentions FROM aggregated_sentiment "
        "WHERE place_id = %s ORDER BY aspect_category;",
        (place["place_id"],),
    )
    aspects = cur.fetchall()

    cur.execute(
        "SELECT summary_text, sources FROM ai_summaries WHERE place_id = %s "
        "ORDER BY generated_at DESC LIMIT 1;",
        (place["place_id"],),
    )
    summary_row = cur.fetchone()

    cur.execute(
        "SELECT text_content FROM reviews_raw WHERE place_id = %s ORDER BY review_id;",
        (place["place_id"],),
    )
    review_texts = [r[0] for r in cur.fetchall()]

    # Last-resort live enrichment — only when this curated place
    # genuinely has zero linked review text, never as a routine
    # enrichment step for every place. Whatever's found is stored
    # immediately (see _fetch_place_knowledge), so this branch only
    # ever runs once per place for the app's whole lifetime — every
    # later call sees non-empty review_texts above and skips
    # straight past this.
    if not review_texts:
        official_website = (place["osm_tags"] or {}).get("website")
        review_texts = _fetch_place_knowledge(conn, place["place_id"], place["name"], official_website)

    # Live recommendation score, replacing the old pre-computed
    # quality_score — computed fresh from this place's CURRENT raw
    # review text, never a stored number. The old score is kept
    # only in a server-side log for comparison, never shown to the
    # traveler, matching the real evidence that it correlates far
    # more weakly with actual outcomes (r=0.171 vs 0.668). Needs no
    # Groq/LLM call at all — the deterministic fallback path relies on
    # exactly this.
    from agent.recommendation_service import predict_recommendation

    tags = place["osm_tags"] or {}
    # A genuine ML-model load failure (missing/corrupted model file) must
    # not crash this whole place lookup — degrade to an honest "no
    # recommendation confidence available" for THIS place rather than
    # losing every other real fact (hours, vibe cluster, AI summary) too.
    # Real, not hypothetical: deterministic_trip_plan() (agent/crew.py)
    # calls this function directly, with no CrewAI tool-wrapper safety
    # net around it — an uncaught exception here would surface as a raw
    # 500 in exactly the "Groq is down" moment this fallback exists for.
    try:
        recommendation = predict_recommendation({
            "name": place["name"],
            "category": place["category"],
            "subcategory": place["subcategory"],
            "opening_hours": place["opening_hours"],
            "price_level": place["price_level"],
            "has_phone": bool(tags.get("phone")),
            "has_website": bool(tags.get("website")),
            "review_count": len(review_texts),
            "reviews": review_texts,
            "distilbert_sentiment_score": place["distilbert_sentiment_score"],
        })
        log.info(
            f"{place['name']!r}: old_quality_score={place['old_quality_score']} "
            f"new_recommendation={recommendation}"
        )
        has_review_text = recommendation["signals"]["has_review_text"]
        recommendation_confidence = (
            round(recommendation["recommendation_probability"] * 100) if has_review_text else None
        )
        recommendation_label = recommendation["label"] if has_review_text else None
        confidence_unavailable_reason = None if has_review_text else "no review text for this place"
    except Exception:
        log.exception(f"{place['name']!r}: recommendation model failed to load/predict — degrading honestly")
        has_review_text = bool(review_texts)
        recommendation_confidence = None
        recommendation_label = None
        # Genuinely different claim from "no review text" — this place may
        # have plenty of reviews; the model itself is what's unavailable.
        # Conflating the two (as an earlier version of this fix did) is
        # exactly the kind of dishonest error message this project has
        # repeatedly had to fix elsewhere (see weather_conditions' own
        # rate_limited/provider_unavailable split).
        confidence_unavailable_reason = "the recommendation model is temporarily unavailable"

    source_urls = []
    summary_text = None
    if summary_row:
        summary_text, sources = summary_row
        source_urls = [s.get("source_url") for s in sources if s.get("source_url")]

    return {
        "name": place["name"],
        "category": place["category"],
        "neighborhood": place["neighborhood"],
        "opening_hours": place["opening_hours"],
        "has_review_text": has_review_text,
        "recommendation_confidence": recommendation_confidence,
        "recommendation_label": recommendation_label,
        "confidence_unavailable_reason": confidence_unavailable_reason,
        "vibe_cluster": place["cluster_label"],
        "aspects": [(a[0], a[1], a[2]) for a in aspects],
        "summary": summary_text,
        "sources": source_urls,
    }


_MAX_PLACE_DETAILS_BATCH = 8


@tool
@_cache_tool_calls
def place_details(place_names: str) -> str:
    """Looks up everything known about one or more Copenhagen places:
    category, neighborhood, opening hours, recommendation confidence,
    vibe cluster, rated aspects, and its RAG-grounded AI summary with
    cited sources.
    Pass ALL the places you need in ONE call, comma-separated (e.g. "Den
    lille Havfrue, Torvehallerne, Nyhavn") — never call this once per
    place, since each separate call resends the whole conversation so far
    and is a real, measured contributor to hitting Groq's per-minute token
    limit. Never state a fact about a place that isn't returned here."""
    names = [n.strip() for n in place_names.split(",") if n.strip()]
    if not names:
        return "No place name given."

    # Hard, deterministic ceiling — not just prompt guidance, which was
    # tried and found unreliable in real testing (the Scout still passed
    # 13 loosely-related candidates through despite explicit instructions
    # to be selective). Each extra place here means a live
    # predict_recommendation() call and, for uncached places, a live
    # web-lookup fetch (agent/tools.py's _fetch_place_knowledge) — real
    # time and, upstream, real tokens once this batch's result goes back
    # into the LLM's context. This caps the worst case regardless of how
    # many candidates the Scout decided to forward.
    truncated = len(names) > _MAX_PLACE_DETAILS_BATCH
    names = names[:_MAX_PLACE_DETAILS_BATCH]

    blocks = []
    with connect() as conn, conn.cursor() as cur:
        for place_name in names:
            detail = _lookup_place_structured(cur, conn, place_name)
            if detail is None:
                blocks.append(f"No place found matching '{place_name}'.")
                continue

            confidence_line = (
                f"Recommendation confidence: {detail['recommendation_confidence']}% "
                f"({detail['recommendation_label']})"
                if detail["recommendation_confidence"] is not None
                else f"Recommendation confidence: not available ({detail['confidence_unavailable_reason']})"
            )
            lines = [
                f"{detail['name']} ({detail['category']}, {detail['neighborhood'] or 'unknown area'})",
                f"Opening hours: {detail['opening_hours'] or 'unknown'}",
                confidence_line,
                f"Vibe cluster: {detail['vibe_cluster'] or 'unclustered'}",
            ]
            if detail["aspects"]:
                lines.append(
                    "Rated aspects: "
                    + ", ".join(f"{a[0]} {a[1]:.1f}/5 ({a[2]} mention(s))" for a in detail["aspects"])
                )
            if detail["summary"]:
                lines.append(f"AI summary: {detail['summary']}")
                if detail["sources"]:
                    lines.append("Sources: " + ", ".join(detail["sources"]))
            else:
                lines.append("AI summary: not available (no linked review text for this place).")
            blocks.append("\n".join(lines))

    if truncated:
        blocks.append(
            f"(Only the first {_MAX_PLACE_DETAILS_BATCH} places were looked up — narrow down "
            "to the ones that genuinely fit before asking again for the rest.)"
        )
    return "\n\n".join(blocks)


@tool
@_cache_tool_calls
def travel_time_estimate(place_names: str) -> str:
    """Estimates straight-line distance and walk/bike time from the
    traveler's starting point to one or more named places. Pass ALL the
    places you need in ONE call, comma-separated — never call this once
    per place, since each separate call resends the whole conversation so
    far and is a real, measured contributor to hitting Groq's per-minute
    token limit. Only useful if a starting point was actually given — if
    it wasn't, says so plainly instead of guessing a location. Not a
    routed transit time (no bus/metro line lookup) — say "roughly" when
    quoting it. If walking would take more than 15 minutes, suggests
    biking or public transit instead of walking."""
    if _trip_start["lat"] is None:
        return "No starting location was given for this trip, so travel time can't be estimated."

    names = [n.strip() for n in place_names.split(",") if n.strip()]
    if not names:
        return "No place name given."

    blocks = []
    with connect() as conn, conn.cursor() as cur:
        for place_name in names:
            resolved = _resolve_place(cur, place_name)
            if not resolved:
                blocks.append(f"No place found matching '{place_name}' to estimate travel time for.")
                continue
            _, name, lat, lon = resolved

            dist_km = haversine_km(_trip_start["lat"], _trip_start["lon"], lat, lon)
            fields = travel_fields(dist_km)

            if fields["travel_note"]:
                blocks.append(
                    f"{name} — from {_trip_start['label']}: roughly {fields['distance_km']:.1f} km "
                    f"straight-line — that's {fields['travel_note']} (~{fields['walk_minutes']} min "
                    f"walk, ~{fields['bike_minutes']} min bike). Otherwise, Copenhagen's Metro/S-train "
                    f"network covers most of the city well — worth checking a real route, since exact "
                    f"bus/train lines aren't available here."
                )
            else:
                blocks.append(
                    f"{name} — from {_trip_start['label']}: roughly {fields['distance_km']:.1f} km "
                    f"straight-line, about {fields['walk_minutes']} min walking or "
                    f"{fields['bike_minutes']} min biking."
                )
    return "\n\n".join(blocks)


_WEATHER_LAT, _WEATHER_LON = 55.6761, 12.5683  # Copenhagen city center, matches ingestion/weather.py
_WEATHER_TZ = ZoneInfo("Europe/Copenhagen")
_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode"
# Open-Meteo's own free forecast endpoint caps at 16 days out — requested
# directly rather than assumed, so "too far to forecast" is driven by what
# the real API actually returns for that date, not a guessed cutoff.
_MAX_FORECAST_DAYS = 16


def _fetch_and_cache_live_weather(conn, d) -> tuple[tuple | None, str | None]:
    """On a weather_daily cache miss, try Open-Meteo live for this one
    date — the archive endpoint for a past date (safe to call for any
    historical date, weather doesn't change), the forecast endpoint for
    today or a near-future date. Returns (row, error_kind): row is
    (temp_max_c, temp_min_c, precip_mm, wind_kph) on success, upserted
    into weather_daily so the next request for the same date hits the
    cache instead of calling the API again — same "fetch once, store it,
    never pay for it twice" pattern search_place_live's Serper-backed
    cousin uses elsewhere in this project. On failure, row is None and
    error_kind is one of "rate_limited" (a real Open-Meteo 429, observed
    live in production) or "provider_unavailable" (network error, 5xx,
    timeout, unparsable response) — found live: without this split, a
    genuine rate-limit hit and "beyond the forecast horizon" were
    reported identically, which is a real, different, and dishonest
    claim about a date only days out. The caller checks the date range
    itself before ever calling this, so that case never reaches here."""
    today = datetime.now(_WEATHER_TZ).date()
    is_past = d < today

    if is_past:
        url = _OPEN_METEO_ARCHIVE_URL
        params = {
            "latitude": _WEATHER_LAT, "longitude": _WEATHER_LON,
            "start_date": d.isoformat(), "end_date": d.isoformat(),
            "daily": _OPEN_METEO_DAILY_FIELDS, "timezone": "Europe/Copenhagen",
        }
    else:
        days_ahead = (d - today).days + 1
        if days_ahead > _MAX_FORECAST_DAYS:
            return None, "out_of_range"  # defense in depth — callers should already check this first
        url = _OPEN_METEO_FORECAST_URL
        params = {
            "latitude": _WEATHER_LAT, "longitude": _WEATHER_LON,
            "forecast_days": days_ahead,
            "daily": _OPEN_METEO_DAILY_FIELDS, "timezone": "Europe/Copenhagen",
        }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        daily = resp.json()["daily"]
    except requests.exceptions.HTTPError as e:
        error_kind = "rate_limited" if e.response is not None and e.response.status_code == 429 else "provider_unavailable"
        log.info(f"Live Open-Meteo call failed for {d} ({error_kind}): {e}")
        return None, error_kind
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        log.info(f"Live Open-Meteo call failed for {d} (provider_unavailable): {e}")
        return None, "provider_unavailable"

    target = d.isoformat()
    if target not in daily["time"]:
        return None, "provider_unavailable"
    i = daily["time"].index(target)
    tmax, tmin, precip, wind, code = (
        daily["temperature_2m_max"][i], daily["temperature_2m_min"][i],
        daily["precipitation_sum"][i], daily["windspeed_10m_max"][i], daily["weathercode"][i],
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_daily (date, temp_max_c, temp_min_c, precip_mm, wind_kph, condition_code, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (date) DO UPDATE SET
                temp_max_c = EXCLUDED.temp_max_c, temp_min_c = EXCLUDED.temp_min_c,
                precip_mm = EXCLUDED.precip_mm, wind_kph = EXCLUDED.wind_kph,
                condition_code = EXCLUDED.condition_code, fetched_at = now();
            """,
            (d, tmax, tmin, precip, wind, code),
        )
    conn.commit()
    return (tmax, tmin, precip, wind), None


def _weather_structured(target_date: str, category: str = "") -> dict:
    """The real weather_conditions lookup, factored out so agent/crew.py's
    deterministic_trip_plan() (used when Groq is confirmed unavailable)
    can get real weather with no LLM call at all. weather_conditions()
    below is now a thin text-formatting wrapper around this.

    Always returns a dict with "ok": bool. On failure, "error_kind" is
    one of "unparsable_date", "out_of_range", "rate_limited", or
    "provider_unavailable" — kept genuinely distinct (not collapsed into
    one generic message) since a real Open-Meteo 429 was once dishonestly
    reported to a traveler as "beyond the forecast horizon" for a date
    only 8 days out. On success, "from_cache" says whether this hit
    weather_daily directly or required a live Open-Meteo call."""
    try:
        # weather_daily.date is a plain SQL date, not timestamptz — the
        # immediate .date() call deliberately discards time/timezone.
        d = datetime.strptime(target_date, "%Y-%m-%d").date()  # noqa: DTZ007
    except ValueError:
        return {"ok": False, "error_kind": "unparsable_date", "target_date": target_date}

    today = datetime.now(_WEATHER_TZ).date()
    # Checked here, before ever attempting a live call, so a genuine
    # out-of-range date and a live call that failed for some OTHER real
    # reason (e.g. Open-Meteo itself rate-limiting Render — a real 429
    # observed live in production) can never be reported as the same
    # thing.
    if d > today and (d - today).days + 1 > _MAX_FORECAST_DAYS:
        return {"ok": False, "error_kind": "out_of_range", "date": d, "days_ahead": (d - today).days}

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT temp_max_c, temp_min_c, precip_mm, wind_kph FROM weather_daily WHERE date = %s;",
            (d,),
        )
        weather_row = cur.fetchone()
        from_cache = weather_row is not None

        if weather_row is None:
            weather_row, error_kind = _fetch_and_cache_live_weather(conn, d)
            if weather_row is None:
                return {"ok": False, "error_kind": error_kind, "date": d}

        sql = (
            "SELECT AVG(vtf.predicted_interest_score) FROM visit_time_forecast vtf "
            "JOIN places p ON p.place_id = vtf.place_id WHERE vtf.forecast_date = %s"
        )
        params = [d]
        if category:
            sql += " AND p.category = %s"
            params.append(category)
        cur.execute(sql, params)
        interest = cur.fetchone()[0]

    temp_max, temp_min, precip, wind = weather_row
    return {
        "ok": True,
        "date": d,
        "from_cache": from_cache,
        "temp_max_c": temp_max,
        "temp_min_c": temp_min,
        "precip_mm": precip,
        "wind_kph": wind,
        "outdoor_interest": interest,
    }


_WEATHER_ERROR_MESSAGES = {
    "out_of_range": (
        "{date} is {days_ahead} days from now — beyond Open-Meteo's real "
        f"{_MAX_FORECAST_DAYS}-day forecast horizon. Treat conditions as unknown "
        "rather than guessing; ask again closer to the date for a real forecast."
    ),
    "rate_limited": (
        "No weather data available for {date} — the weather provider (Open-Meteo) is "
        "temporarily rate-limiting requests, not that the date is out of range. Treat "
        "conditions as unknown for now; a retry shortly should succeed."
    ),
    "provider_unavailable": (
        "No weather data available for {date} — a live lookup was attempted but the "
        "weather provider didn't return it (a network issue or temporary outage, not that "
        "the date is out of range). Treat conditions as unknown rather than guessing."
    ),
}


@tool
@_cache_tool_calls
def weather_conditions(target_date: str, category: str = "") -> str:
    """Weather and outdoor-interest conditions for one date (YYYY-MM-DD),
    optionally scoped to a place category (restaurant, cafe, hotel,
    landmark). Combines real Open-Meteo weather with the weather-aware
    forecasting model's Outdoor Interest Index (0-100). Covers historical
    dates (from 2025-01-01), today, and up to 16 days ahead (Open-Meteo's
    real forecast horizon) — live, on demand, not just whatever a periodic
    batch job happened to have already stored. Dates further out than that
    say so honestly instead of guessing."""
    result = _weather_structured(target_date, category)

    if not result["ok"]:
        if result["error_kind"] == "unparsable_date":
            return f"Could not parse '{target_date}' as a date (expected YYYY-MM-DD)."
        return _WEATHER_ERROR_MESSAGES[result["error_kind"]].format(
            date=result["date"], days_ahead=result.get("days_ahead")
        )

    d, temp_max, temp_min, precip, wind, interest = (
        result["date"], result["temp_max_c"], result["temp_min_c"],
        result["precip_mm"], result["wind_kph"], result["outdoor_interest"],
    )
    lines = [f"{d}: high {temp_max}°C / low {temp_min}°C, {precip}mm precipitation, {wind}km/h wind."]
    if interest is not None:
        scope = f" for {category}" if category else ""
        lines.append(f"Predicted outdoor interest index{scope}: {interest:.1f}/100.")
    else:
        # Real weather above can now come from a live Open-Meteo call (up to
        # 16 days out), but the Outdoor Interest Index is a separate, older
        # signal — trained on and only precomputed for whatever window
        # pipeline/timeseries/forecast_interest.py's periodic batch run last
        # covered. The two windows aren't the same; don't blur them.
        lines.append("No outdoor-interest forecast available for that date (outside the periodically-updated forecast batch's date range).")
    return "\n".join(lines)
