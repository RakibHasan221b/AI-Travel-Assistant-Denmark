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


class TripPlanResponse(BaseModel):
    itinerary: str


@app.post("/trip-plan", response_model=TripPlanResponse)
def trip_plan(body: TripPlanRequest):
    if not body.request.strip():
        raise HTTPException(400, "request must not be empty")
    try:
        itinerary = plan_trip(body.request, body.target_date)
    except Exception as e:
        log.exception("Crew run failed")
        raise HTTPException(500, f"Trip planning failed: {e}") from e
    return TripPlanResponse(itinerary=itinerary)
