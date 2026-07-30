"""Phase 12 — Trip Planner page: calls the Phase 11 FastAPI /trip-plan
endpoint (the CrewAI crew itself doesn't run in-process here — Streamlit's
rerun-on-every-interaction model is exactly why Phase 11 put the crew behind
its own FastAPI service instead, see docs/architecture.md)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_api_url

# Deployed on cloud infra that may not run in Copenhagen's timezone — this
# app is Copenhagen-scoped (weather_daily, forecasts), so "today" needs to
# mean Copenhagen-local today, not the server's own timezone.
COPENHAGEN_TODAY = datetime.now(ZoneInfo("Europe/Copenhagen")).date()

st.set_page_config(page_title="Trip Planner — AI Denmark Explorer", page_icon="🗺️", layout="centered")
st.title("🗺️ Trip Planner")
st.caption(
    "Three agents collaborate on this: a Place Scout finds candidates, a Conditions Analyst "
    "checks real weather, and a Concierge writes the final recommendation grounded in what "
    "the other two actually found."
)

with st.form("trip_form"):
    request_text = st.text_area(
        "What are you looking for?",
        placeholder="I want a cozy quiet cafe to work from and one landmark to visit nearby, I like calm places",
        height=100,
    )
    # Explicit field, not left to the LLM to parse out of free text — the
    # underlying search_places/top_quality_places tools already take a
    # neighborhood filter, this just makes it a reliable, visible input
    # instead of something you have to know to type inline.
    area = st.selectbox(
        "Area in Copenhagen (optional)",
        ["Any area", "Vesterbro", "Norrebro", "Osterbro", "Frederiksberg", "Indre By"],
    )
    start_location = st.text_input(
        "Starting from (optional)",
        placeholder="Copenhagen Central Station, or your hotel name/address",
        help="Used to estimate walking/biking distance to recommended places. Skip it and no travel time will be shown.",
    )
    target_date = st.date_input("Target date", value=COPENHAGEN_TODAY + timedelta(days=1))
    submitted = st.form_submit_button("Plan my trip", type="primary")

if submitted:
    if not request_text.strip():
        st.warning("Describe what you're looking for first.")
    else:
        full_request = request_text if area == "Any area" else f"In {area}: {request_text}"
        with st.spinner("Planning your trip — this runs three real agents and real tool calls, usually 1-2 minutes..."):
            try:
                resp = requests.post(
                    f"{get_api_url()}/trip-plan",
                    json={
                        "request": full_request,
                        "target_date": target_date.isoformat(),
                        "start_location": start_location.strip(),
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                st.success("Here's the plan:")
                st.write(resp.json()["itinerary"])
            except requests.exceptions.RequestException as e:
                detail = ""
                if getattr(e, "response", None) is not None:
                    try:
                        detail = e.response.json().get("detail", "")
                    except ValueError:
                        detail = e.response.text
                st.error(f"Trip planning failed: {detail or e}")
