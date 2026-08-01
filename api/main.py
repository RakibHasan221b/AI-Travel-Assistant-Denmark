"""Phase 11 — thin FastAPI service in front of the CrewAI trip-planning crew.

Only needed starting here: the crew needs stable session/process state that
fights Streamlit's rerun-on-every-interaction model (see docs/architecture.md).
Deliberately thin — one real endpoint, no auth/persistence layer, matching
this project's single-user pilot scope.
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.crew import plan_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="AI Denmark Explorer — Trip Planner")


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
    quality_score: float | None = None
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
    renders this as cards (quality score, distance, weather) instead of
    asking the LLM to "format nicely," which costs zero extra tokens and is
    actually more reliable than hoping the model produces good layout in
    free text — see docs/technique_map.md for the full before/after story."""

    places: list[PlaceRecommendation]
    weather_summary: str
    overall_note: str = ""


@app.post("/trip-plan", response_model=TripPlanResponse)
def trip_plan(body: TripPlanRequest):
    if not body.request.strip():
        raise HTTPException(400, "request must not be empty")
    try:
        result = plan_trip(body.request, body.target_date, body.start_location)
    except Exception as e:
        log.exception("Crew run failed")
        raise HTTPException(500, f"Trip planning failed: {e}") from e
    return TripPlanResponse(**result)
