"""Backfill places.neighborhood via point-in-polygon matching against real
administrative boundaries, not Nominatim reverse geocoding.

Root cause of the "unknown area" gap: neighborhood came from OSM's
addr:suburb tag (osm_common.py), which most individual place nodes never
carry — same underlying pattern as the missing-address gap (address info
usually lives on the containing building, not the point). Reverse-geocoding
every place through Nominatim would work but costs ~35 min at its 1 req/s
policy (1,896 places). This is faster and, arguably, more correct: a
one-time bulk download of real district boundaries (11 polygons total),
then plain local geometry — no rate limit, no ongoing network dependency
after the first run.

Two sources, because Copenhagen's 10 official districts don't cover the
whole pilot bbox:
- opendata.dk / Kobenhavns Kommune: the 10 official Bydele (districts)
  covering Copenhagen Municipality itself.
- DAWA (Danmarks Adressers Web API, dataforsyningen.dk — the Danish
  government's own address/administrative-boundary API): Frederiksberg is
  a separate, independent municipality enclaved inside Copenhagen's bbox,
  not one of the 10 Bydele, so it needs its own boundary from a different
  source. Tried Overpass first for this (same free OSM ecosystem the rest
  of ingestion uses) — got the exact flaky/overloaded 504 this project's
  own architecture notes already warned about for Overpass, so used DAWA
  instead, which answered reliably.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv
from shapely.geometry import Point, shape

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("neighborhood_backfill")

CACHE_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "boundaries" / "districts.json"
BYDELE_URL = "https://wfs-kbhkort.kk.dk/k101/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=k101:bydel&outputFormat=json&SRSNAME=EPSG:4326"
DAWA_URL = "https://api.dataforsyningen.dk/kommuner"


def fetch_districts() -> list[dict]:
    bydele = requests.get(BYDELE_URL, timeout=30)
    bydele.raise_for_status()
    frederiksberg = requests.get(DAWA_URL, params={"q": "Frederiksberg", "format": "geojson"}, timeout=20)
    frederiksberg.raise_for_status()

    districts = [
        {"name": f["properties"]["navn"], "geometry": f["geometry"]}
        for f in bydele.json()["features"]
    ]
    districts += [
        {"name": f["properties"]["navn"], "geometry": f["geometry"]}
        for f in frederiksberg.json()["features"]
    ]
    return districts


def load_districts() -> list[dict]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    districts = fetch_districts()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(districts), encoding="utf-8")
    return districts


def build_polygons(districts: list[dict]) -> list[tuple[str, object]]:
    return [(d["name"], shape(d["geometry"])) for d in districts]


def find_district(lon: float, lat: float, polygons: list[tuple[str, object]]) -> str | None:
    point = Point(lon, lat)
    for name, polygon in polygons:
        if polygon.contains(point):
            return name
    return None


def main():
    db_url = os.environ["DATABASE_URL"]
    polygons = build_polygons(load_districts())
    log.info(f"Loaded {len(polygons)} district boundaries")

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT place_id, lat, lon FROM places WHERE neighborhood IS NULL;")
            rows = cur.fetchall()
        log.info(f"{len(rows)} places missing a neighborhood")

        matched = 0
        with conn.cursor() as cur:
            for place_id, lat, lon in rows:
                name = find_district(lon, lat, polygons)
                if name:
                    cur.execute(
                        "UPDATE places SET neighborhood = %s, updated_at = now() WHERE place_id = %s;",
                        (name, place_id),
                    )
                    matched += 1

            run_id = f"neighborhood-backfill-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            cur.execute(
                """
                INSERT INTO pipeline_runs (run_id, stage, completed_at, status, records_processed, notes)
                VALUES (%s, 'neighborhood_backfill', now(), 'completed', %s, %s);
                """,
                (run_id, len(rows), f"matched={matched}, unmatched={len(rows) - matched}"),
            )
        conn.commit()

    log.info(f"Done. {matched}/{len(rows)} places matched to a district ({len(rows) - matched} outside all known boundaries).")


if __name__ == "__main__":
    main()
