"""Web-search enrichment for places with thin or no linked text.

Real problem this fixes: some places (e.g. the Little Mermaid statue) exist
in `places` but have zero linked `reviews_raw` rows — their only text is a
bare OSM name — so Phase 8's RAG summarizer has nothing to ground a summary
in, and the trip-planning agent's `place_details` tool honestly reports "no
information" instead of inventing one (see docs/architecture_explainer.html
for the live example that surfaced this).

Deliberately NOT another LLM call or a live agent tool: this is a plain
search-and-store script, same shape as osm_live_lookup.py and
wikivoyage_descriptions.py. Storing web-search text as ordinary
`reviews_raw` rows means Phase 8's existing RAG pipeline picks it up on its
next run with zero changes there — enrichment and summarization stay two
separate, single-purpose steps rather than one script doing both.

Search provider: Serper.dev (Google Search API) — 2,500 free searches, no
credit card required, comfortably covers backfilling the places that
actually have thin data (a few hundred, not the full 1,896).
"""

import argparse
import logging
import os
from datetime import UTC, datetime

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web_enrichment")

SERPER_URL = "https://google.serper.dev/search"
MAX_RESULTS_PER_PLACE = 3
MIN_SNIPPET_CHARS = 40

# Social/video platforms return search snippets too thin or off-topic to
# ground a summary in (a caption, a comment fragment) — skip them in favor
# of encyclopedic/official/editorial sources.
LOW_SIGNAL_DOMAINS = (
    "facebook.com", "instagram.com", "tiktok.com", "youtube.com",
    "twitter.com", "x.com", "pinterest.com", "reddit.com",
)


def search_web(query: str) -> list[dict]:
    resp = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"},
        json={"q": query},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("organic", [])


def filter_results(results: list[dict]) -> list[dict]:
    kept = []
    for r in results:
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        if len(snippet) < MIN_SNIPPET_CHARS:
            continue
        if any(domain in link for domain in LOW_SIGNAL_DOMAINS):
            continue
        kept.append(r)
        if len(kept) >= MAX_RESULTS_PER_PLACE:
            break
    return kept


def find_thin_places(conn, limit: int) -> list[dict]:
    """Places with zero linked reviews_raw rows — landmarks first, since
    that's the category the Little Mermaid case showed is worst-affected
    (OSM rarely tags a landmark's history/description, unlike a
    restaurant's cuisine or a hotel's amenities)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.place_id::text, p.name, p.category
            FROM places p
            WHERE NOT EXISTS (SELECT 1 FROM reviews_raw r WHERE r.place_id = p.place_id)
            ORDER BY (p.category = 'landmark') DESC, p.name
            LIMIT %s;
            """,
            (limit,),
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def enrich_place(cur, place_id: str, place_name: str, existing: set) -> int:
    results = filter_results(search_web(f"{place_name} Copenhagen"))
    inserted = 0
    for r in results:
        source_url = r["link"]
        if (place_id, source_url) in existing:
            continue
        cur.execute(
            """
            INSERT INTO reviews_raw (place_id, source_type, source_id, source_url,
                                      text_content, run_id, raw_payload)
            VALUES (%s, 'web_search', %s, %s, %s, %s, %s)
            RETURNING review_id;
            """,
            (
                place_id,
                r.get("title", place_name),
                source_url,
                r["snippet"],
                f"web-enrichment-{datetime.now(UTC).strftime('%Y%m%d')}",
                psycopg.types.json.Json(r),
            ),
        )
        review_id = cur.fetchone()[0]
        # Also record in place_mentions, not just reviews_raw.place_id:
        # embed_places.py (Phase 6) builds each place's OWN search-ranking
        # embedding by joining through place_mentions, not by reading
        # reviews_raw.place_id directly — skipping this would fix RAG
        # summaries (Phase 8 reads reviews_raw.place_id directly) but leave
        # the actual search-ranking problem that motivated this tool
        # unfixed. match_method='manual': the place_id here is already a
        # confirmed exact lookup, not a fuzzy geographic/name guess.
        cur.execute(
            """
            INSERT INTO place_mentions (review_id, place_id, match_method, confidence)
            VALUES (%s, %s, 'manual', 1.0);
            """,
            (review_id, place_id),
        )
        existing.add((place_id, source_url))
        inserted += 1
    return inserted


def run(place_ids_and_names: list[tuple[str, str]], db_url: str) -> tuple[int, int]:
    places_enriched = rows_inserted = 0
    with psycopg.connect(db_url, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            # Idempotency guard, same natural-key pattern as
            # wikivoyage_descriptions.py: (place_id, source_url) — re-running
            # this against the same place must not duplicate rows.
            cur.execute("SELECT place_id::text, source_url FROM reviews_raw WHERE source_type = 'web_search';")
            existing = {(row[0], row[1]) for row in cur.fetchall()}

            for place_id, place_name in place_ids_and_names:
                n = enrich_place(cur, place_id, place_name, existing)
                if n:
                    places_enriched += 1
                    rows_inserted += n
                    log.info(f"  {place_name!r}: {n} source(s) added")
                else:
                    log.info(f"  {place_name!r}: nothing new found")

            run_id = f"web-enrich-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            cur.execute(
                """
                INSERT INTO pipeline_runs (run_id, stage, completed_at, status, records_processed, notes)
                VALUES (%s, 'web_enrichment', now(), 'completed', %s, %s);
                """,
                (run_id, rows_inserted, f"places_enriched={places_enriched}"),
            )
        conn.commit()
    return places_enriched, rows_inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", help="Enrich one place by exact name (e.g. 'The Little Mermaid')")
    parser.add_argument("--backfill-thin", action="store_true", help="Enrich places with zero linked reviews_raw rows")
    parser.add_argument("--limit", type=int, default=50, help="Max places to enrich in --backfill-thin mode")
    args = parser.parse_args()

    db_url = os.environ["DATABASE_URL"]

    if args.place:
        with psycopg.connect(db_url, connect_timeout=15) as conn, conn.cursor() as cur:
            # Exact (case-insensitive) match first — an ILIKE substring-only
            # match is ambiguous when near-duplicate OSM entries exist for
            # the same real-world place (found live: "Den lille Havfrue" vs
            # "Den lille havfrue #2" vs a genuinely different statue,
            # "Den Genmodificerede Lille Havfrue" — substring matching alone
            # picked the wrong one).
            cur.execute("SELECT place_id::text, name FROM places WHERE lower(name) = lower(%s) LIMIT 1;", (args.place,))
            row = cur.fetchone()
            if not row:
                cur.execute("SELECT place_id::text, name FROM places WHERE name ILIKE %s ORDER BY name LIMIT 1;", (f"%{args.place}%",))
                row = cur.fetchone()
        if not row:
            log.error(f"No place found matching {args.place!r}")
            return
        targets = [(row[0], row[1])]
    elif args.backfill_thin:
        with psycopg.connect(db_url, connect_timeout=15) as conn:
            thin = find_thin_places(conn, args.limit)
        log.info(f"{len(thin)} places with zero linked text found (limit {args.limit})")
        targets = [(p["place_id"], p["name"]) for p in thin]
    else:
        parser.error("pass --place NAME or --backfill-thin")
        return

    places_enriched, rows_inserted = run(targets, db_url)
    log.info(f"Done. {places_enriched}/{len(targets)} places got new sources, {rows_inserted} reviews_raw rows added.")


if __name__ == "__main__":
    main()
