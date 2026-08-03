"""Phase 12 — Trip Planner page: calls the Phase 11 FastAPI /trip-plan
endpoint (the CrewAI crew itself doesn't run in-process here — Streamlit's
rerun-on-every-interaction model is exactly why Phase 11 put the crew behind
its own FastAPI service instead, see docs/architecture.md).

Result rendering, explained (this replaced a single free-text paragraph):
the Concierge agent now returns structured data (TripPlanOutput — see
agent/crew.py) instead of prose. This page turns that data into cards,
the same way app/pages/1_Explore.py already renders search results — same
visual language across the app, and this formatting step costs zero LLM
tokens, since it's plain Python running after the agent is already done.
Asking the LLM to "format nicely" in free text would be less reliable (it's
just guessing at layout) and would cost tokens on every single request for
something deterministic code does for free.

Result also lives in st.session_state so it survives Streamlit reruns —
switching the Area dropdown, or navigating to another page and back, no
longer wipes it. Before this, the result only existed inside the
`if submitted:` block's local scope, which Streamlit throws away on the
next rerun (a form submission button's True/False state doesn't persist).
"""

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
    "Two agents collaborate on this: a Place Scout finds candidates, and a Concierge checks "
    "real weather and writes the final recommendation grounded in what the Scout actually found."
)

if "trip_plan_result" not in st.session_state:
    st.session_state.trip_plan_result = None


def render_trip_plan(result: dict) -> None:
    st.info(f"🌤️ {result['weather_summary']}")

    for place in result["places"]:
        with st.container(border=True):
            header_col, score_col = st.columns([4, 1])
            header_col.markdown(f"**{place['name']}** — {place['category']}, {place['neighborhood']}")
            if place["quality_score"] is not None:
                score_col.metric("Quality", f"{place['quality_score']:.0f}/100")

            meta_bits = []
            if place.get("vibe_cluster"):
                meta_bits.append(f"Vibe: *{place['vibe_cluster']}*")
            if place.get("opening_hours"):
                meta_bits.append(f"🕐 {place['opening_hours']}")
            if place.get("near_place") and place.get("near_distance_km") is not None:
                meta_bits.append(f"🚶 {place['near_distance_km']:.2f} km from {place['near_place']}")
            if place.get("distance_km") is not None:
                if place.get("travel_note"):
                    meta_bits.append(f"📍 {place['distance_km']:.1f} km from your start — {place['travel_note']}")
                else:
                    time_bits = []
                    if place.get("walk_minutes") is not None:
                        time_bits.append(f"{place['walk_minutes']} min walk")
                    if place.get("bike_minutes") is not None:
                        time_bits.append(f"{place['bike_minutes']} min bike")
                    time_str = f" ({' / '.join(time_bits)})" if time_bits else ""
                    meta_bits.append(f"📍 {place['distance_km']:.1f} km from your start{time_str}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))

            st.write(place["why_recommended"])
            if place.get("summary"):
                st.caption(place["summary"])
            if place.get("sources"):
                st.caption("Sources: " + ", ".join(place["sources"]))

    if result.get("overall_note"):
        st.markdown("##### Overall")
        st.write(result["overall_note"])


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
                # Only overwrite on success — a failed new attempt shouldn't
                # wipe out a plan that already worked.
                st.session_state.trip_plan_result = resp.json()
            except requests.exceptions.RequestException as e:
                detail = ""
                if getattr(e, "response", None) is not None:
                    try:
                        detail = e.response.json().get("detail", "")
                    except ValueError:
                        detail = e.response.text
                st.error(f"Trip planning failed: {detail or e}")

if st.session_state.trip_plan_result:
    st.success("Here's the plan:")
    render_trip_plan(st.session_state.trip_plan_result)
