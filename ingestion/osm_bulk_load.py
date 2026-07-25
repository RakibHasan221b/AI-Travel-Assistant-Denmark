"""Phase 1 (primary path) — bulk-load Copenhagen places from a Geofabrik
regional OSM extract, processed locally with pyosmium.

Why bulk instead of the live Overpass API for the initial load: the free
public Overpass servers are shared, rate-limited, and proved unreliable for
a full-category pull in testing (timeouts, inconsistent results even on
identical repeat requests). Geofabrik extracts are static files with no
rate limit — download once, filter locally, no flakiness. Geofabrik refreshes
its extracts roughly daily, so re-running this periodically keeps data current
without depending on a live query API for bulk volume.

The live Overpass path (osm_live_lookup.py) stays in the project for small,
targeted, on-demand queries, which testing showed work fine at that scale.
"""

import logging
import os
from pathlib import Path

import osmium
import requests
from dotenv import load_dotenv

from osm_common import BBOX, in_bbox, to_row, upsert

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("osm_bulk_load")

GEOFABRIK_URL = "https://download.geofabrik.de/europe/denmark-latest.osm.pbf"
CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "osm"
CACHE_PATH = CACHE_DIR / "denmark-latest.osm.pbf"


def download_extract() -> Path:
    if CACHE_PATH.exists():
        size_mb = CACHE_PATH.stat().st_size / 1_000_000
        log.info(f"Using cached extract: {CACHE_PATH} ({size_mb:.0f} MB)")
        return CACHE_PATH

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading Denmark extract from Geofabrik (one-time, cached after)...")
    with requests.get(GEOFABRIK_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(CACHE_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1_000_000:.0f} / {total / 1_000_000:.0f} MB ({pct:.0f}%)", end="")
        print()
    log.info(f"Downloaded to {CACHE_PATH}")
    return CACHE_PATH


class PlaceHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rows: dict[str, dict] = {}  # osm_id -> row, dedupes automatically

    def node(self, n):
        if not n.location.valid():
            return
        lat, lon = n.location.lat, n.location.lon
        if not in_bbox(lat, lon):
            return
        row = to_row("node", n.id, n.tags, lat, lon)
        if row:
            self.rows[row["osm_id"]] = row

    def way(self, w):
        locations = [nd.location for nd in w.nodes if nd.location.valid()]
        if not locations:
            return
        lat = sum(loc.lat for loc in locations) / len(locations)
        lon = sum(loc.lon for loc in locations) / len(locations)
        if not in_bbox(lat, lon):
            return
        row = to_row("way", w.id, w.tags, lat, lon)
        if row:
            self.rows[row["osm_id"]] = row


def main():
    db_url = os.environ["DATABASE_URL"]
    extract_path = download_extract()

    log.info(f"Processing extract for bbox {BBOX} (this reads the whole Denmark file locally, no network calls)...")
    handler = PlaceHandler()
    handler.apply_file(str(extract_path), locations=True)

    rows = list(handler.rows.values())
    log.info(f"Found {len(rows)} named restaurants/cafes/hotels/landmarks in the pilot area")

    inserted, updated = upsert(rows, db_url, stage="osm_bulk_load")
    log.info(f"Done. Inserted {inserted} new places, updated {updated} existing.")


if __name__ == "__main__":
    main()
