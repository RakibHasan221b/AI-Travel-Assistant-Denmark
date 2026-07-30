"""Offline-safe: importing agent.tools must not touch the network or DB —
only calling a tool does. DATABASE_URL isn't read until a tool actually runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import (
    haversine_km,
    place_details,
    search_places,
    set_trip_start,
    top_quality_places,
    travel_time_estimate,
    weather_conditions,
)

EXPECTED_TOOLS = [
    search_places, top_quality_places, place_details, travel_time_estimate, weather_conditions,
]


def test_tools_have_names_and_descriptions():
    for t in EXPECTED_TOOLS:
        assert t.name
        assert t.description


def test_weather_conditions_rejects_unparsable_date_without_hitting_the_network():
    result = weather_conditions.run(target_date="not-a-date")
    assert "Could not parse" in result


def test_travel_time_estimate_without_a_start_point_says_so_without_hitting_the_network():
    set_trip_start(None, None, "")
    result = travel_time_estimate.run(place_name="Den lille Havfrue")
    assert "No starting location was given" in result


def test_haversine_km_known_distance():
    # Copenhagen Central Station to the Little Mermaid statue, roughly 3.3 km
    # straight-line — a real, checkable distance, not an arbitrary number.
    dist = haversine_km(55.6726, 12.5646, 55.6929, 12.5993)
    assert 2.5 < dist < 4.0


def test_haversine_km_zero_distance_for_same_point():
    assert haversine_km(55.68, 12.57, 55.68, 12.57) == 0.0
