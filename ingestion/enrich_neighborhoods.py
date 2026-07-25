"""One-time enrichment: assign an approximate neighborhood to places OSM left
unlabeled (addr:suburb is rarely tagged on Danish POI nodes — see Phase 1
notes in the plan).

This is a deliberate approximation, not authoritative district boundaries:
rough bounding boxes for Copenhagen's well-known bydele (districts), matched
locally against lat/lon already in the database. No live API calls, no new
downloads. Good enough for search/filtering in a pilot; if precise official
boundaries ever matter, replace with a boundary-polygon join against OSM's
admin_level=10 relations.
"""

import logging
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("enrich_neighborhoods")

# (name, south, west, north, east) — approximate, ordered most-specific first
NEIGHBORHOODS = [
    ("Indre By", 55.674, 12.565, 55.686, 12.595),
    ("Vesterbro", 55.665, 12.530, 55.680, 12.565),
    ("Norrebro", 55.686, 12.540, 55.706, 12.565),
    ("Osterbro", 55.695, 12.565, 55.715, 12.600),
    ("Frederiksberg", 55.670, 12.500, 55.690, 12.535),
]


def assign_neighborhood(lat: float, lon: float) -> str | None:
    for name, s, w, n, e in NEIGHBORHOODS:
        if s <= lat <= n and w <= lon <= e:
            return name
    return None


def main():
    db_url = os.environ["DATABASE_URL"]
    updated = 0
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT place_id, lat, lon FROM places WHERE neighborhood IS NULL;")
            rows = cur.fetchall()
            log.info(f"{len(rows)} places missing neighborhood")

            for place_id, lat, lon in rows:
                neighborhood = assign_neighborhood(lat, lon)
                if neighborhood:
                    cur.execute(
                        "UPDATE places SET neighborhood = %s, updated_at = now() WHERE place_id = %s",
                        (neighborhood, place_id),
                    )
                    updated += 1
        conn.commit()
    log.info(f"Assigned an approximate neighborhood to {updated}/{len(rows)} places")


if __name__ == "__main__":
    main()
