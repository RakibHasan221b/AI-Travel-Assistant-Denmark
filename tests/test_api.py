"""Offline-safe: importing api.main must not touch the network, DB, or LLM —
it only builds the FastAPI app and its routes. Actually running /trip-plan
requires DATABASE_URL, GROQ_API_KEY, and live Groq/DB access, so that's
exercised manually (see README), not in this offline suite."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import litellm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


def test_expected_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/trip-plan" in paths
    assert "/explore" in paths
    assert "/stats" in paths


def test_cors_middleware_is_registered():
    # Real problem this guards: the Next.js frontend (Phase 14) calls this
    # API directly from the browser, cross-origin — if this middleware ever
    # got removed or reordered out, every request from the deployed
    # frontend would silently fail CORS preflight with no obvious cause.
    middleware_classes = [m.cls for m in app.user_middleware]
    assert CORSMiddleware in middleware_classes


def test_cors_allows_vercel_preview_subdomains():
    cors_middleware = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    regex = cors_middleware.kwargs["allow_origin_regex"]
    assert re.match(regex, "https://ai-denmark-explorer-web-git-feat-x-rakib.vercel.app")
    assert re.match(regex, "https://ai-denmark-explorer-web.vercel.app")


_FALLBACK_RESULT = {
    "places": [],
    "weather_summary": "sunny",
    "overall_note": "Our AI trip-planning assistant is temporarily unavailable...",
}


def test_trip_plan_falls_back_to_deterministic_results_on_a_groq_rate_limit(monkeypatch):
    # Real problem this guards: before this fallback existed, ANY plan_trip()
    # exception — including a routine Groq TPM rate limit, not a bug —
    # became a raw HTTP 500. Groq is the reasoning/orchestration layer, not
    # the ML recommendation engine, so losing it shouldn't mean losing the
    # whole response.
    def _raise(*a, **k):
        raise litellm.RateLimitError(message="rate limited", llm_provider="groq", model="llama-3.3-70b-versatile")

    monkeypatch.setattr(api_main, "plan_trip", _raise)
    monkeypatch.setattr(api_main, "deterministic_trip_plan", lambda *a, **k: _FALLBACK_RESULT)

    client = TestClient(app)
    resp = client.post("/trip-plan", json={"request": "cozy cafe", "target_date": "2026-08-15"})
    assert resp.status_code == 200
    assert resp.json()["overall_note"] == _FALLBACK_RESULT["overall_note"]


def test_trip_plan_falls_back_on_a_persisted_malformed_tool_call(monkeypatch):
    # The malformed-tool-call retry lives inside plan_trip() itself (one
    # cheap in-crew retry); if it fails AGAIN, that BadRequestError
    # propagates all the way here — this is the final safety net, not a
    # duplicate of plan_trip's own retry.
    def _raise(*a, **k):
        raise litellm.BadRequestError(
            message='{"error": {"code": "tool_use_failed"}}', llm_provider="groq", model="llama-3.3-70b-versatile"
        )

    monkeypatch.setattr(api_main, "plan_trip", _raise)
    monkeypatch.setattr(api_main, "deterministic_trip_plan", lambda *a, **k: _FALLBACK_RESULT)

    client = TestClient(app)
    resp = client.post("/trip-plan", json={"request": "cozy cafe", "target_date": "2026-08-15"})
    assert resp.status_code == 200


def test_trip_plan_still_returns_500_on_a_genuine_unexpected_error(monkeypatch):
    # The fallback is a narrow, deliberate safety net for known Groq
    # failure modes — NOT a blanket catch-all. A genuine application bug
    # (or any exception type not in _GROQ_UNAVAILABLE_ERRORS) must still
    # surface as a real error, not get silently swallowed into a 200.
    def _raise(*a, **k):
        raise RuntimeError("something genuinely broke")

    monkeypatch.setattr(api_main, "plan_trip", _raise)

    client = TestClient(app)
    resp = client.post("/trip-plan", json={"request": "cozy cafe", "target_date": "2026-08-15"})
    assert resp.status_code == 500
