"""Live data — on-demand single-place lookups via Nominatim, OSM's official
search service.

Not the raw Overpass API: testing showed the free community Overpass mirrors
are flaky even on narrow queries (identical requests succeeding once, then
failing three times in a row). Nominatim is different — it's the official
search service behind openstreetmap.org itself, explicitly designed for
exactly this use case (look up one place by name), and its usage policy
explicitly prohibits the opposite use case (bulk POI dumps over an area) —
which is fine, that's what osm_bulk_load.py is for. Free forever, no API key,
1 request/second limit (fine for on-demand single lookups, not for scanning
an area).

If a category+area live search is ever needed (not a single named place),
Geoapify's Places API (free tier, OSM-based, needs an API key) is the better
tool for that — noted here rather than built, since it's not needed yet.
"""

import logging
import os
import time

import requests
from dotenv import load_dotenv

from osm_common import classify, to_row, upsert

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("osm_live_lookup")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "ai-denmark-explorer/0.1 (Copenhagen pilot, personal project)"}

_last_call = 0.0


def _respect_rate_limit():
    # Nominatim's usage policy: max 1 request/second.
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    _last_call = time.time()


def lookup_place(name: str, city: str = "Copenhagen") -> dict | None:
    """Look up one named place. Returns a normalized row, or None if not found."""
    _respect_rate_limit()
    params = {
        "q": f"{name}, {city}, Denmark",
        "format": "jsonv2",
        "extratags": 1,
        "limit": 1,
    }
    resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        log.info(f"No Nominatim result for '{name}'")
        return None

    result = results[0]
    tags = result.get("extratags") or {}
    tags["name"] = tags.get("name") or result.get("name") or name
    lat, lon = float(result["lat"]), float(result["lon"])
    osm_type = result.get("osm_type", "node")
    osm_id = result.get("osm_id")

    row = to_row(osm_type, osm_id, tags, lat, lon)
    if not row:
        log.info(f"'{name}' found but tags don't classify as a supported category: {tags}")
    return row


def lookup_and_store(name: str, db_url: str) -> bool:
    row = lookup_place(name)
    if not row:
        return False
    inserted, updated = upsert([row], db_url, stage="osm_live_lookup")
    return inserted + updated > 0


if __name__ == "__main__":
    # Demo: look up one real, well-known Copenhagen place on demand.
    db_url = os.environ["DATABASE_URL"]
    found = lookup_and_store("Rundetaarn", db_url)
    log.info(f"Live lookup demo {'succeeded' if found else 'found nothing usable'}.")
