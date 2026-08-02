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

import math
import os
import time
from datetime import datetime

import psycopg
import requests
from crewai.tools import tool
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

load_dotenv()

# Set once per plan_trip() call, before crew.kickoff() — a module-level
# global, not a per-request object, because crewai's @tool functions are
# plain module-level functions with no natural place to inject per-request
# state. Same "fine at this project's demo scale" trade-off as the DB
# connection pattern above: a real concurrent-request server would need
# this threaded through properly instead. See agent/crew.py's plan_trip().
_trip_start: dict = {"lat": None, "lon": None, "label": None}


def set_trip_start(lat: float | None, lon: float | None, label: str) -> None:
    _trip_start["lat"], _trip_start["lon"], _trip_start["label"] = lat, lon, label


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


@tool
def search_places(query: str, category: str = "", neighborhood: str = "", limit: int | str = 5) -> str:
    """Semantic search over Copenhagen places (pgvector). Use this to find
    candidate places matching a vibe or description, e.g. "cozy quiet cafe
    good for working". Optionally filter by category (restaurant, cafe,
    hotel, landmark, bar) and/or neighborhood."""
    model = get_embed_model()
    query_embedding = next(model.embed([query]))
    sql = """
        SELECT name, category, neighborhood, opening_hours,
               1 - (embedding <=> %(qvec)s) AS similarity
        FROM places
        WHERE embedding IS NOT NULL
    """
    params = {"qvec": query_embedding, "limit": _coerce_limit(limit)}
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

    if not rows:
        return "No matching places found."
    return "\n".join(
        f"- {r['name']} ({r['category']}, {r['neighborhood'] or 'unknown area'}), "
        f"similarity {r['similarity']:.2f}, hours: {r['opening_hours'] or 'unknown'}"
        for r in rows
    )


@tool
def search_places_near(anchor_place: str, category: str = "", limit: int | str = 5) -> str:
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

        sql = "SELECT name, category, neighborhood, opening_hours, lat, lon FROM places WHERE place_id != %(anchor_id)s"
        params = {"anchor_id": anchor_id}
        if category:
            sql += " AND category = %(category)s"
            params["category"] = category
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    for r in rows:
        r["distance_km"] = haversine_km(anchor_lat, anchor_lon, r["lat"], r["lon"])
    rows.sort(key=lambda r: r["distance_km"])
    rows = [r for r in rows if r["distance_km"] <= MAX_NEARBY_KM][:limit]

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


@tool
def search_place_live(query: str) -> str:
    """Looks up a real place by name via a live OpenStreetMap/Nominatim
    search — use this ONLY after search_places, search_places_near, and
    top_quality_places all found nothing for a named place the traveler
    asked about. This checks whether the place genuinely exists in
    Copenhagen even though it isn't in our curated dataset.

    A result from this tool has NO quality score, vibe cluster, or AI
    summary — it was never scored or evaluated. Say so plainly if you use
    it; never present it with the same confidence as a place found by the
    other tools."""
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
    return (
        f"LIVE LOOKUP RESULT (not in our curated dataset — no quality score, "
        f"vibe cluster, or AI summary available): {name}, {address}."
    )


@tool
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


@tool
def place_details(place_name: str) -> str:
    """Looks up everything known about one Copenhagen place by (partial)
    name: category, neighborhood, opening hours, quality score, vibe
    cluster, rated aspects, and its RAG-grounded AI summary with cited
    sources. Use this before finalizing a recommendation for a specific
    place — never state a fact about a place that isn't returned here."""
    with connect() as conn, conn.cursor() as cur:
        resolved = _resolve_place(cur, place_name)
        if not resolved:
            return f"No place found matching '{place_name}'."
        place_id = resolved[0]

        cur.execute(
            """
            SELECT p.place_id, p.name, p.category, p.neighborhood, p.opening_hours,
                   m.predicted_value AS quality_score, c.label AS cluster_label
            FROM places p
            LEFT JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
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

    quality_line = (
        f"Quality score: {place['quality_score']:.1f}/100"
        if place["quality_score"] is not None
        else "Quality score: not available"
    )
    lines = [
        f"{place['name']} ({place['category']}, {place['neighborhood'] or 'unknown area'})",
        f"Opening hours: {place['opening_hours'] or 'unknown'}",
        quality_line,
        f"Vibe cluster: {place['cluster_label'] or 'unclustered'}",
    ]
    if aspects:
        lines.append(
            "Rated aspects: "
            + ", ".join(f"{a[0]} {a[1]:.1f}/5 ({a[2]} mention(s))" for a in aspects)
        )
    if summary_row:
        summary_text, sources = summary_row
        lines.append(f"AI summary: {summary_text}")
        source_urls = [s.get("source_url") for s in sources if s.get("source_url")]
        if source_urls:
            lines.append("Sources: " + ", ".join(source_urls))
    else:
        lines.append("AI summary: not available (no linked review text for this place).")
    return "\n".join(lines)


@tool
def travel_time_estimate(place_name: str) -> str:
    """Estimates straight-line distance and walk/bike time from the
    traveler's starting point to a named place. Only useful if a starting
    point was actually given — if it wasn't, says so plainly instead of
    guessing a location. Not a routed transit time (no bus/metro line
    lookup) — say "roughly" when quoting it. If walking would take more
    than 15 minutes, suggests biking or public transit instead of walking."""
    if _trip_start["lat"] is None:
        return "No starting location was given for this trip, so travel time can't be estimated."

    with connect() as conn, conn.cursor() as cur:
        resolved = _resolve_place(cur, place_name)
    if not resolved:
        return f"No place found matching '{place_name}' to estimate travel time for."
    _, _, lat, lon = resolved

    dist_km = haversine_km(_trip_start["lat"], _trip_start["lon"], lat, lon)
    fields = travel_fields(dist_km)

    if fields["travel_note"]:
        return (
            f"From {_trip_start['label']}: roughly {fields['distance_km']:.1f} km straight-line — "
            f"that's {fields['travel_note']} (~{fields['walk_minutes']} min walk, "
            f"~{fields['bike_minutes']} min bike). Otherwise, Copenhagen's Metro/S-train network "
            f"covers most of the city well — worth checking a real route, since exact bus/train "
            f"lines aren't available here."
        )
    return (
        f"From {_trip_start['label']}: roughly {fields['distance_km']:.1f} km straight-line, "
        f"about {fields['walk_minutes']} min walking or {fields['bike_minutes']} min biking."
    )


@tool
def weather_conditions(target_date: str, category: str = "") -> str:
    """Weather and outdoor-interest conditions for one date (YYYY-MM-DD),
    optionally scoped to a place category (restaurant, cafe, hotel,
    landmark). Combines real Open-Meteo weather with the weather-aware
    forecasting model's Outdoor Interest Index (0-100). Covers both
    historical dates (from 2025-01-01) and the current ~7-day forecast
    window. If the date falls outside the stored range, says so honestly
    instead of guessing."""
    try:
        # weather_daily.date is a plain SQL date, not timestamptz — the
        # immediate .date() call deliberately discards time/timezone.
        d = datetime.strptime(target_date, "%Y-%m-%d").date()  # noqa: DTZ007
    except ValueError:
        return f"Could not parse '{target_date}' as a date (expected YYYY-MM-DD)."

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT temp_max_c, temp_min_c, precip_mm, wind_kph FROM weather_daily WHERE date = %s;",
            (d,),
        )
        weather_row = cur.fetchone()

        if weather_row is None:
            cur.execute("SELECT MIN(date), MAX(date) FROM weather_daily;")
            min_d, max_d = cur.fetchone()
            return (
                f"No weather data stored for {d}. Available range is {min_d} to {max_d} — "
                "pick a date in that window, or treat conditions as unknown rather than guessing."
            )

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
    lines = [f"{d}: high {temp_max}°C / low {temp_min}°C, {precip}mm precipitation, {wind}km/h wind."]
    if interest is not None:
        scope = f" for {category}" if category else ""
        lines.append(f"Predicted outdoor interest index{scope}: {interest:.1f}/100.")
    else:
        lines.append("No outdoor-interest forecast available for that date (likely a historical date, or the 7-day forecast window has moved on).")
    return "\n".join(lines)
