"""Shared logic between the bulk (Geofabrik) and live (Overpass) OSM ingestion
paths — tag classification, address building, and the upsert into `places`.
Both paths produce the same row shape, so this stays the one place that shape
is defined.
"""

import json
import logging
from datetime import datetime, timezone

import psycopg

log = logging.getLogger("osm_common")

# Central Copenhagen pilot area (Indre By, Vesterbro, Norrebro, Frederiksberg core) —
# south, west, north, east. Narrower than the full municipality on purpose; widen
# once there's a reason to (the bulk path can easily cover more once this works).
BBOX = (55.66, 12.52, 55.71, 12.60)

# OSM tag -> (category, subcategory) — subcategory is overridden for restaurants (cuisine)
TAG_TO_CATEGORY = {
    ("amenity", "restaurant"): ("restaurant", None),
    ("amenity", "cafe"): ("cafe", None),
    ("tourism", "hotel"): ("hotel", None),
    ("tourism", "attraction"): ("landmark", None),
}


def in_bbox(lat: float, lon: float) -> bool:
    s, w, n, e = BBOX
    return s <= lat <= n and w <= lon <= e


def classify(tags: dict):
    for (key, value), category in TAG_TO_CATEGORY.items():
        if tags.get(key) == value:
            return category
    return None


def build_address(tags: dict) -> str | None:
    parts = [tags.get("addr:street"), tags.get("addr:housenumber")]
    street = " ".join(p for p in parts if p)
    city_parts = [tags.get("addr:postcode"), tags.get("addr:city")]
    city = " ".join(p for p in city_parts if p)
    full = ", ".join(p for p in [street, city] if p)
    return full or None


def to_row(osm_type: str, osm_id: int, tags: dict, lat: float, lon: float) -> dict | None:
    name = tags.get("name")
    if not name:
        return None
    classification = classify(tags)
    if not classification:
        return None
    category, subcategory = classification
    if category == "restaurant":
        subcategory = tags.get("cuisine")
    return {
        "osm_id": f"{osm_type}/{osm_id}",
        "name": name,
        "category": category,
        "subcategory": subcategory,
        "lat": lat,
        "lon": lon,
        "address": build_address(tags),
        "neighborhood": tags.get("addr:suburb"),
        "opening_hours": tags.get("opening_hours"),
        "osm_tags": json.dumps(dict(tags)),
    }


UPSERT_SQL = """
    INSERT INTO places (osm_id, name, category, subcategory, lat, lon,
                         address, neighborhood, opening_hours, osm_tags)
    VALUES (%(osm_id)s, %(name)s, %(category)s, %(subcategory)s, %(lat)s, %(lon)s,
            %(address)s, %(neighborhood)s, %(opening_hours)s, %(osm_tags)s)
    ON CONFLICT (osm_id) DO UPDATE SET
        name = EXCLUDED.name,
        category = EXCLUDED.category,
        subcategory = EXCLUDED.subcategory,
        lat = EXCLUDED.lat,
        lon = EXCLUDED.lon,
        address = EXCLUDED.address,
        neighborhood = EXCLUDED.neighborhood,
        opening_hours = EXCLUDED.opening_hours,
        osm_tags = EXCLUDED.osm_tags,
        updated_at = now()
    RETURNING (xmax = 0) AS inserted;
"""


def upsert(rows: list[dict], db_url: str, stage: str) -> tuple[int, int]:
    if not rows:
        return 0, 0
    inserted = updated = 0
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(UPSERT_SQL, row)
                if cur.fetchone()[0]:
                    inserted += 1
                else:
                    updated += 1

            run_id = f"cph-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            cur.execute(
                """
                INSERT INTO pipeline_runs (run_id, stage, completed_at, status, records_processed, notes)
                VALUES (%s, %s, now(), 'completed', %s, %s)
                """,
                (run_id, stage, len(rows), f"inserted={inserted}, updated={updated}"),
            )
        conn.commit()
    log.info(f"Upsert done: {inserted} new, {updated} updated (run {run_id})")
    return inserted, updated
