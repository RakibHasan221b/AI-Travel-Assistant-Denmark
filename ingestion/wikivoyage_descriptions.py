"""Phase 3 — pull real editorial descriptions from Wikivoyage's Copenhagen
district articles and link them to places already in the database.

Free, no key (MediaWiki API), paced to its "no more than one page per 30s"
guideline. Wikivoyage listings ({{see|...}}, {{eat|...}}, etc.) carry their
own lat/long, which lets matching lean on coordinate proximity rather than
name-string fuzziness alone — more reliable than either signal on its own.
"""

import logging
import math
import os
import re
import time
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wikivoyage_descriptions")

CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "wikivoyage"

API_URL = "https://en.wikivoyage.org/w/api.php"
HEADERS = {"User-Agent": "ai-denmark-explorer/0.1 (Copenhagen pilot, personal project)"}
MAIN_ARTICLE = "Copenhagen"

LISTING_BLOCK_RE = re.compile(
    r"\{\{\s*(see|eat|drink|do|sleep|buy)\s*(.*?)\n\}\}", re.DOTALL | re.IGNORECASE
)
FIELD_RE = re.compile(r"\|\s*(\w+)\s*=\s*([^|]*)")
CONTENT_RE = re.compile(r"content\s*=\s*(.*)", re.DOTALL)

MATCH_DISTANCE_METERS = 100

# A Wikivoyage listing type only proves a match against a place of a plausible
# category — proximity alone isn't enough (a hotel and a nearby restaurant can
# both fall within 100m of each other, and are not the same venue).
LISTING_TYPE_TO_CATEGORIES = {
    "see": {"landmark"},
    "do": {"landmark"},
    "eat": {"restaurant"},
    "drink": {"cafe"},
    "sleep": {"hotel"},
    "buy": set(),  # no matching category in our schema; never fuzzy-matched
}


def api_get(params: dict, max_attempts: int = 3) -> dict:
    for attempt in range(1, max_attempts + 1):
        resp = requests.get(API_URL, params={**params, "format": "json"}, headers=HEADERS, timeout=20)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 30))
            log.warning(f"429 rate limited, waiting {wait}s (attempt {attempt}/{max_attempts})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def get_district_articles() -> list[str]:
    data = api_get(
        {"action": "query", "titles": MAIN_ARTICLE, "prop": "links", "plnamespace": 0, "pllimit": 500}
    )
    page = next(iter(data["query"]["pages"].values()))
    titles = [l["title"] for l in page.get("links", [])]
    return [t for t in titles if t.startswith(f"{MAIN_ARTICLE}/")]


def get_wikitext(title: str) -> str:
    cache_path = CACHE_DIR / f"{title.replace('/', '_')}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    data = api_get(
        {"action": "query", "titles": title, "prop": "revisions", "rvprop": "content", "rvslots": "main"}
    )
    page = next(iter(data["query"]["pages"].values()))
    revisions = page.get("revisions")
    text = revisions[0]["slots"]["main"]["*"] if revisions else ""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_listings(wikitext: str) -> list[dict]:
    listings = []
    for match in LISTING_BLOCK_RE.finditer(wikitext):
        listing_type, body = match.group(1).lower(), match.group(2)
        fields = {k.lower(): v.strip() for k, v in FIELD_RE.findall(body)}
        content_match = CONTENT_RE.search(body)
        content = content_match.group(1).strip() if content_match else ""
        # strip a trailing '}}' if the greedy content capture picked one up
        content = re.sub(r"\}\}\s*$", "", content).strip()
        if not fields.get("name") or not content:
            continue
        try:
            lat = float(fields["lat"]) if fields.get("lat") else None
            lon = float(fields["long"]) if fields.get("long") else None
        except ValueError:
            lat = lon = None
        listings.append(
            {
                "type": listing_type,
                "name": fields["name"],
                "lat": lat,
                "lon": lon,
                "content": content,
            }
        )
    return listings


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_place(listing: dict, places: list[dict]) -> tuple[str, str, float] | None:
    """Returns (place_id, match_method, confidence) or None."""
    allowed_categories = LISTING_TYPE_TO_CATEGORIES.get(listing["type"], set())

    # Exact name match still requires a plausible category — a same-named
    # coincidence across an unrelated category isn't a real match either.
    norm_listing_name = normalize_name(listing["name"])
    for p in places:
        if p["category"] not in allowed_categories:
            continue
        if normalize_name(p["name"]) == norm_listing_name:
            return p["place_id"], "exact_name", 1.0

    if listing["lat"] is not None and listing["lon"] is not None and allowed_categories:
        best, best_dist = None, MATCH_DISTANCE_METERS
        for p in places:
            if p["category"] not in allowed_categories:
                continue
            d = haversine_m(listing["lat"], listing["lon"], p["lat"], p["lon"])
            if d < best_dist:
                best, best_dist = p, d
        if best:
            confidence = max(0.5, 1 - best_dist / MATCH_DISTANCE_METERS)
            return best["place_id"], "fuzzy", round(confidence, 2)

    return None


def main():
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, connect_timeout=15) as conn, conn.cursor() as cur:
        cur.execute("SELECT place_id::text, name, lat, lon, category FROM places;")
        places = [
            {"place_id": r[0], "name": r[1], "lat": r[2], "lon": r[3], "category": r[4]}
            for r in cur.fetchall()
        ]
    log.info(f"Loaded {len(places)} places for matching")

    districts = get_district_articles()
    log.info(f"Found {len(districts)} Wikivoyage district articles")

    all_listings = []
    for i, title in enumerate(districts):
        log.info(f"Fetching '{title}' ({i + 1}/{len(districts)})...")
        try:
            wikitext = get_wikitext(title)
        except requests.exceptions.RequestException as e:
            log.warning(f"  skipping '{title}' after retries failed: {e}")
            continue
        listings = parse_listings(wikitext)
        for listing in listings:
            listing["source_title"] = title
        all_listings.extend(listings)
        log.info(f"  {len(listings)} listings parsed")
        if i < len(districts) - 1:
            time.sleep(30)  # Wikivoyage's own guideline: no more than one page per 30s

    log.info(f"Total listings parsed: {len(all_listings)}")

    linked = unlinked = skipped = 0
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            # Idempotency guard: this script has no natural DB constraint to
            # rely on (unlike osm_common.py's ON CONFLICT (osm_id) or
            # weather.py's ON CONFLICT (date)), so re-running it against the
            # same cached articles would otherwise insert duplicate
            # reviews_raw rows every time — (source_id, source_url) is the
            # natural key for one Wikivoyage listing.
            cur.execute("SELECT source_id, source_url FROM reviews_raw WHERE source_type = 'wikivoyage';")
            existing = {(row[0], row[1]) for row in cur.fetchall()}

            for listing in all_listings:
                source_url = f"https://en.wikivoyage.org/wiki/{listing['source_title'].replace(' ', '_')}"
                if (listing["name"], source_url) in existing:
                    skipped += 1
                    continue

                match = match_place(listing, places)
                place_id = match[0] if match else None
                cur.execute(
                    """
                    INSERT INTO reviews_raw (place_id, source_type, source_id, source_url,
                                              text_content, run_id, raw_payload)
                    VALUES (%s, 'wikivoyage', %s, %s, %s, %s, %s)
                    RETURNING review_id;
                    """,
                    (
                        place_id,
                        listing["name"],
                        source_url,
                        listing["content"],
                        "wikivoyage-initial-load",
                        psycopg.types.json.Json(listing),
                    ),
                )
                review_id = cur.fetchone()[0]
                if match:
                    place_id, method, confidence = match
                    cur.execute(
                        """
                        INSERT INTO place_mentions (review_id, place_id, match_method, confidence)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (review_id, place_id, method, confidence),
                    )
                    linked += 1
                else:
                    unlinked += 1
        conn.commit()
    log.info(f"Skipped {skipped} already-ingested listings.")

    log.info(f"Done. {linked} listings linked to places, {unlinked} stored unlinked.")


if __name__ == "__main__":
    main()
