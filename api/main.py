"""Phase 11 — thin FastAPI service in front of the CrewAI trip-planning crew.

Only needed starting here: the crew needs stable session/process state that
fights Streamlit's rerun-on-every-interaction model (see docs/architecture.md).
Deliberately thin — one real endpoint, no auth/persistence layer, matching
this project's single-user pilot scope.
"""

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.crew import _trip_plan_from_cached_results, deterministic_trip_plan, plan_trip
from agent.tools import connect, get_embed_model
from api.ranking import DEFAULT_POOL_SIZE, rank_explore_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="AI Denmark Explorer — Trip Planner")

# Needed once a browser-based client (the Next.js frontend, Phase 14) calls
# this API directly cross-origin — see docs/technique_map.md. No credentials
# (cookies/auth) on this endpoint, so a plain origin allowlist is enough; no
# need for the stricter allow_credentials=True configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    # Vercel gives every branch/PR its own *.vercel.app preview subdomain,
    # which a fixed allowlist can't cover — match the whole project's
    # preview + production domains with one regex instead.
    allow_origin_regex=r"https://ai-denmark-explorer.*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class TripPlanRequest(BaseModel):
    request: str
    target_date: str
    start_location: str = ""


class PlaceRecommendation(BaseModel):
    name: str
    category: str
    neighborhood: str = "unknown area"
    opening_hours: str | None = None
    recommendation_confidence: float | None = None
    recommendation_label: str | None = None
    vibe_cluster: str | None = None
    summary: str | None = None
    sources: list[str] = []
    distance_km: float | None = None
    walk_minutes: int | None = None
    bike_minutes: int | None = None
    travel_note: str | None = None
    near_place: str | None = None
    near_distance_km: float | None = None
    why_recommended: str


class TripPlanResponse(BaseModel):
    """Mirrors agent.crew.TripPlanOutput field-for-field, but declared
    independently rather than imported — this module shouldn't need to know
    about crew.py's internal schema class, only the plain dict plan_trip()
    returns. Structured on purpose, not a single prose string: the frontend
    renders this as cards (recommendation confidence, distance, weather)
    instead of asking the LLM to "format nicely," which costs zero extra
    tokens and is actually more reliable than hoping the model produces
    good layout in free text — see docs/technique_map.md for the full
    before/after story."""

    places: list[PlaceRecommendation]
    weather_summary: str
    overall_note: str = ""


def _groq_unavailable_errors() -> tuple[type[Exception], ...]:
    """Groq/litellm failures that plan_trip() itself can't resolve — a
    TPM/TPD rate limit (plan_trip deliberately never retries these — see
    its own docstring), a malformed tool call that persisted through
    plan_trip's one in-crew retry, no server capacity, or a transient
    outage/timeout. Groq is the reasoning/orchestration layer, not the ML
    recommendation engine — losing it shouldn't mean returning a raw 500
    when the real place data and recommendation-confidence model (which
    need no Groq call at all) can still answer the request. Anything NOT
    in this tuple (a genuine bug, a DB outage, etc.) still surfaces as a
    real 500 below — this is a deliberate, narrow safety net for known
    provider failure modes, not a blanket catch-all that would hide real
    application bugs.

    Imports litellm lazily, on first request, not at module load —
    `import litellm` alone costs ~120MB RSS (measured), and crewai
    already only imports it lazily itself (confirmed: bare `import
    crewai` never touches sys.modules for litellm). A module-level import
    here would silently undo the memory work that got this process
    comfortably under Render's 512MB limit, paying that cost on every
    process start whether or not /trip-plan is ever hit."""
    import litellm

    return (
        litellm.RateLimitError,
        litellm.BadRequestError,
        litellm.APIConnectionError,
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
        litellm.Timeout,
    )


def _is_groq_unavailable_error(e: Exception) -> bool:
    """True for a Groq/litellm failure plan_trip() itself can't resolve.

    Matches by module prefix as well as the explicit tuple above —
    pyproject.toml's `agent` extra pins no crewai/litellm version, so
    Render re-resolves both to whatever is newest on every deploy. litellm
    has reshuffled which concrete exception subclass a given provider
    error maps to across versions before; a version bump that changes the
    subclass for Groq's rate-limit response would otherwise miss the
    isinstance() tuple and let a plain litellm-originated failure escape
    as a raw 500 instead of reaching the fallback below. crewai's own
    executor uses this identical `__module__.startswith("litellm")` check
    for the same reason (see crew_agent_executor.py's _invoke_loop_react)."""
    return isinstance(e, _groq_unavailable_errors()) or type(e).__module__.startswith("litellm")


@app.post("/trip-plan", response_model=TripPlanResponse)
def trip_plan(body: TripPlanRequest):
    if not body.request.strip():
        raise HTTPException(400, "request must not be empty")
    try:
        result = plan_trip(body.request, body.target_date, body.start_location)
    except Exception as e:
        if not _is_groq_unavailable_error(e):
            log.exception("Crew run failed")
            raise HTTPException(500, f"Trip planning failed: {e}") from e
        log.warning(f"Groq unavailable ({type(e).__name__}), falling back to deterministic results: {e}")
        # Try reusing whatever the failed crew run already computed
        # (place_details results with real recommendation_confidence,
        # etc.) before discarding it for a brand-new whole-sentence
        # search — see _trip_plan_from_cached_results' own docstring for
        # the real bug this fixes. Only genuinely falls through to
        # deterministic_trip_plan() when the cache has nothing usable.
        result = _trip_plan_from_cached_results(body.request, body.target_date, body.start_location)
        if result is None:
            result = deterministic_trip_plan(body.request, body.target_date, body.start_location)
    return TripPlanResponse(**result)


class ExplorePlaceResult(BaseModel):
    name: str
    category: str
    neighborhood: str = "unknown area"
    opening_hours: str | None = None
    similarity: float
    quality_score: float | None = None
    vibe_cluster: str | None = None
    summary: str | None = None
    sources: list[str] = []


class ExploreResponse(BaseModel):
    results: list[ExplorePlaceResult]


@app.get("/explore", response_model=ExploreResponse)
def explore(
    q: str = Query(..., description="What to search for"),
    category: str = Query(""),
    neighborhood: str = Query(""),
    limit: int = Query(5, ge=1, le=20),
):
    """Semantic search (pgvector) over Copenhagen places, ported from
    app/pages/1_Explore.py, plus a deterministic reranking layer
    (api/ranking.py) applied after retrieval. Real problem that fix
    addresses: raw "ORDER BY embedding <=> qvec LIMIT n" treats every one
    of the n nearest vectors as equally relevant — a real query for
    "little mermaid" correctly put "Den lille Havfrue" first (similarity
    0.47) but padded the rest of the list with unrelated hotels/
    restaurants/statues down around 0.20-0.30, just because they were the
    next-nearest embeddings. Retrieval now pulls a wider candidate pool
    (DEFAULT_POOL_SIZE) than what's actually shown, rank_explore_
    candidates() combines that raw similarity with real lexical name
    matching and category-intent detection, and only genuinely relevant
    candidates survive — which can mean fewer than `limit` results, or
    zero, not padding to fill the page. quality_score/vibe_cluster stay
    exactly as they were — this only changes which candidates are
    returned and in what order, never what's shown about each one. Uses
    agent.tools.connect()/get_embed_model() rather than its own copies —
    see agent/tools.py's docstring for why (shared process, shared
    fastembed singleton)."""
    if not q.strip():
        raise HTTPException(400, "q must not be empty")

    model = get_embed_model()
    query_embedding = next(model.embed([q]))

    sql = """
        SELECT p.place_id, p.name, p.category, p.subcategory, p.neighborhood, p.opening_hours,
               1 - (p.embedding <=> %(qvec)s) AS similarity,
               m.predicted_value AS quality_score,
               c.label AS vibe_cluster
        FROM places p
        LEFT JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
        LEFT JOIN place_clusters pc ON pc.place_id = p.place_id
        LEFT JOIN clusters c ON c.cluster_id = pc.cluster_id
        WHERE p.embedding IS NOT NULL
    """
    # Pool is wider than what's actually returned — the real candidate a
    # query actually means (e.g. the exact-name match) isn't always in the
    # raw embedding-only top few, so the reranker needs a generous pool to
    # promote it from.
    params: dict = {"qvec": query_embedding, "limit": DEFAULT_POOL_SIZE}
    if category:
        sql += " AND p.category = %(category)s"
        params["category"] = category
    if neighborhood.strip():
        sql += " AND p.neighborhood ILIKE %(neighborhood)s"
        params["neighborhood"] = f"%{neighborhood.strip()}%"
    sql += " ORDER BY p.embedding <=> %(qvec)s LIMIT %(limit)s;"

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]

        ranked = rank_explore_candidates(q, rows, final_limit=limit)

        results = []
        for r in ranked:
            cur.execute(
                "SELECT summary_text, sources FROM ai_summaries WHERE place_id = %s "
                "ORDER BY generated_at DESC LIMIT 1;",
                (r["place_id"],),
            )
            summary_row = cur.fetchone()
            summary_text = summary_row[0] if summary_row else None
            source_urls = (
                [s.get("source_url") for s in summary_row[1] if s.get("source_url")]
                if summary_row
                else []
            )
            results.append(
                ExplorePlaceResult(
                    name=r["name"],
                    category=r["category"],
                    neighborhood=r["neighborhood"] or "unknown area",
                    opening_hours=r["opening_hours"],
                    similarity=r["similarity"],
                    quality_score=r["quality_score"],
                    vibe_cluster=r["vibe_cluster"],
                    summary=summary_text,
                    sources=source_urls,
                )
            )

    return ExploreResponse(results=results)


class NeighborhoodQuality(BaseModel):
    neighborhood: str
    avg_quality_score: float
    n_places: int


class CategoryQuality(BaseModel):
    category: str
    avg_quality_score: float
    n_places: int


class VibeClusterSize(BaseModel):
    label: str
    n_places: int


class RatedAspect(BaseModel):
    aspect_category: str
    mean_score: float
    total_mentions: int


class ForecastPoint(BaseModel):
    forecast_date: str
    category: str
    avg_interest: float


class StatsResponse(BaseModel):
    """All 5 of app/pages/3_Stats_Dashboard.py's aggregate queries in one
    payload, matching that page's "whole dashboard loads at once" UX rather
    than fragmenting one screen's data across 5 endpoints. Forecast stays in
    long format exactly as SQL returns it — pivoting to wide format for the
    chart is a frontend concern (web/lib/stats.ts), not this endpoint's."""

    quality_by_neighborhood: list[NeighborhoodQuality]
    quality_by_category: list[CategoryQuality]
    vibe_cluster_sizes: list[VibeClusterSize]
    rated_aspects: list[RatedAspect]
    outdoor_interest_forecast: list[ForecastPoint]


@app.get("/stats", response_model=StatsResponse)
def stats():
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(p.neighborhood, 'unassigned') AS neighborhood,
                   AVG(m.predicted_value) AS avg_quality_score, COUNT(*) AS n_places
            FROM places p
            JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
            GROUP BY p.neighborhood
            HAVING COUNT(*) >= 10
            ORDER BY avg_quality_score DESC;
            """
        )
        columns = [d.name for d in cur.description]
        quality_by_neighborhood = [dict(zip(columns, r)) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT p.category, AVG(m.predicted_value) AS avg_quality_score, COUNT(*) AS n_places
            FROM places p
            JOIN ml_predictions m ON m.place_id = p.place_id AND m.target = 'quality_score'
            GROUP BY p.category
            ORDER BY avg_quality_score DESC;
            """
        )
        columns = [d.name for d in cur.description]
        quality_by_category = [dict(zip(columns, r)) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT c.label, COUNT(*) AS n_places
            FROM clusters c
            JOIN place_clusters pc ON pc.cluster_id = c.cluster_id
            GROUP BY c.label
            ORDER BY n_places DESC;
            """
        )
        columns = [d.name for d in cur.description]
        vibe_cluster_sizes = [dict(zip(columns, r)) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT aspect_category, AVG(avg_score) AS mean_score, SUM(num_mentions) AS total_mentions
            FROM aggregated_sentiment
            GROUP BY aspect_category
            ORDER BY mean_score DESC;
            """
        )
        columns = [d.name for d in cur.description]
        rated_aspects = [dict(zip(columns, r)) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT vtf.forecast_date, p.category, AVG(vtf.predicted_interest_score) AS avg_interest
            FROM visit_time_forecast vtf
            JOIN places p ON p.place_id = vtf.place_id
            GROUP BY vtf.forecast_date, p.category
            ORDER BY vtf.forecast_date;
            """
        )
        columns = [d.name for d in cur.description]
        outdoor_interest_forecast = [
            {**dict(zip(columns, r)), "forecast_date": str(r[0])} for r in cur.fetchall()
        ]

    return StatsResponse(
        quality_by_neighborhood=quality_by_neighborhood,
        quality_by_category=quality_by_category,
        vibe_cluster_sizes=vibe_cluster_sizes,
        rated_aspects=rated_aspects,
        outdoor_interest_forecast=outdoor_interest_forecast,
    )
