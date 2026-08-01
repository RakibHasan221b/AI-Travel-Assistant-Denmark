"""Offline-safe: importing api.main must not touch the network, DB, or LLM —
it only builds the FastAPI app and its routes. Actually running /trip-plan
requires DATABASE_URL, GROQ_API_KEY, and live Groq/DB access, so that's
exercised manually (see README), not in this offline suite."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.middleware.cors import CORSMiddleware

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
