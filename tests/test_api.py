"""Offline-safe: importing api.main must not touch the network, DB, or LLM —
it only builds the FastAPI app and its routes. Actually running /trip-plan
requires DATABASE_URL, GROQ_API_KEY, and live Groq/DB access, so that's
exercised manually (see README), not in this offline suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.main import app


def test_expected_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/trip-plan" in paths
