"""Phase 8 — RAG-grounded place summaries.

Retrieval: pgvector similarity search (Phase 6) over each place's own linked
reviews_raw rows only — never another place's text — so the "sources" cited
alongside a summary are genuinely what grounded it. Generation: GPT-4o via
LangChain, the one place in the stack this model is used (see
docs/architecture.md's ML/rules-before-LLM-calls principle).

Only places with at least one linked review/description are summarized —
skipping the rest is a deliberate scoping decision, not an oversight: with
nothing retrieved there is nothing to ground a summary in, and generating one
anyway would be exactly the hallucination RAG exists to prevent.
"""

import logging
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm" / "prompts"))
from rag_summary import RAG_SUMMARY_PROMPT

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_summaries")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-4o"
SNIPPETS_PER_PLACE = 5
SNIPPET_CHARS = 800

# Approximate OpenAI GPT-4o pricing (per 1M tokens, post-2024 price cut).
# Update these if pricing changes — cost_usd_est is an estimate, not a bill.
INPUT_PRICE_PER_1M = 2.50
OUTPUT_PRICE_PER_1M = 10.00


def load_candidate_places(conn) -> list[dict]:
    """Places with at least one linked review/description and no summary yet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.place_id, p.name, p.category, p.neighborhood
            FROM places p
            WHERE EXISTS (SELECT 1 FROM reviews_raw r WHERE r.place_id = p.place_id)
              AND NOT EXISTS (SELECT 1 FROM ai_summaries s WHERE s.place_id = p.place_id);
            """
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def retrieve_snippets(conn, model: SentenceTransformer, place: dict) -> list[dict]:
    """Vector search scoped to this place's own reviews_raw rows only."""
    query_text = f"{place['name']} {place['category']} atmosphere food service value"
    query_embedding = model.encode(query_text)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT review_id, text_content, source_type, source_url,
                   1 - (embedding <=> %(qvec)s) AS similarity
            FROM reviews_raw
            WHERE place_id = %(place_id)s AND embedding IS NOT NULL
            ORDER BY embedding <=> %(qvec)s
            LIMIT %(k)s;
            """,
            {"qvec": query_embedding, "place_id": place["place_id"], "k": SNIPPETS_PER_PLACE},
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def fetch_aspect_facts(conn, place_id) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT aspect_category, avg_score, num_mentions
            FROM aggregated_sentiment
            WHERE place_id = %s
            ORDER BY aspect_category;
            """,
            (place_id,),
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def format_aspect_facts(rows: list[dict]) -> str:
    if not rows:
        return "No rated aspects yet."
    return "\n".join(
        f"- {r['aspect_category']}: {r['avg_score']:.1f}/5 ({r['num_mentions']} mention(s))"
        for r in rows
    )


def format_snippets(snippets: list[dict]) -> str:
    return "\n".join(
        f"[{i + 1}] ({s['source_type']}) {s['text_content'][:SNIPPET_CHARS]}"
        for i, s in enumerate(snippets)
    )


def estimate_cost(usage: dict) -> float:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return round(
        (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        + (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M,
        6,
    )


def summarize(chain, place: dict, snippets: list[dict], aspect_rows: list[dict]) -> tuple[str, dict]:
    response = chain.invoke(
        {
            "place_name": place["name"],
            "category": place["category"],
            "neighborhood": place["neighborhood"] or "unknown area",
            "aspect_facts": format_aspect_facts(aspect_rows),
            "snippets": format_snippets(snippets),
        }
    )
    usage = response.usage_metadata or {}
    return response.content.strip(), usage


def main():
    db_url = os.environ["DATABASE_URL"]
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)
    chain = ChatPromptTemplate.from_template(RAG_SUMMARY_PROMPT) | llm

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        register_vector(conn)
        places = load_candidate_places(conn)
        log.info(f"{len(places)} places with grounding data and no summary yet")

        with conn.cursor() as cur:
            for i, place in enumerate(places):
                snippets = retrieve_snippets(conn, embed_model, place)
                aspect_rows = fetch_aspect_facts(conn, place["place_id"])
                summary_text, usage = summarize(chain, place, snippets, aspect_rows)
                cost = estimate_cost(usage)

                sources = [
                    {
                        "review_id": str(s["review_id"]),
                        "source_type": s["source_type"],
                        "source_url": s["source_url"],
                    }
                    for s in snippets
                ]
                cur.execute(
                    """
                    INSERT INTO ai_summaries (place_id, summary_text, model_used,
                                               prompt_version, sources, cost_usd_est)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        place["place_id"],
                        summary_text,
                        DEFAULT_MODEL,
                        "v1",
                        psycopg.types.json.Json(sources),
                        cost,
                    ),
                )
                if (i + 1) % 20 == 0:
                    conn.commit()
                    log.info(f"  [{i + 1}/{len(places)}] summarized so far")
        conn.commit()

    log.info(f"Done. {len(places)} places summarized.")


if __name__ == "__main__":
    main()
