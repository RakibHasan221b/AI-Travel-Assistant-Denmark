"""A/B test: Phase 8's RAG summary prompt (v1, paragraph) vs a v2 variant
(verdict-first). Reuses Phase 8's own retrieval/generation building blocks
(pipeline/rag/generate_summaries.py) so both variants see identical
retrieved snippets and aspect facts for a given place — the prompt template
is the only thing that varies. Scored deterministically (ab_scoring.py), not
via another LLM call. Results logged to experiment_results for every place
in the sample, so the raw outputs stay inspectable, not just the tally.

Run: python pipeline/experiments/ab_test_rag_summaries.py [--sample-size N]
Needs the same env as Phase 8: DATABASE_URL, OPENAI_API_KEY (rag extra).
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm" / "prompts"))
from ab_scoring import pick_winner, score_summary
from generate_summaries import (
    DEFAULT_MODEL,
    EMBED_MODEL_NAME,
    fetch_aspect_facts,
    format_aspect_facts,
    format_snippets,
    retrieve_snippets,
)
from rag_summary import RAG_SUMMARY_PROMPT
from rag_summary_v2 import RAG_SUMMARY_PROMPT_V2

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ab_test_rag_summaries")

EXPERIMENT_NAME = "rag_summary_prompt_v1_vs_v2"


def load_sample_places(conn, sample_size: int) -> list[dict]:
    """Places with grounding data, already-summarized ones first (Phase 8
    already vetted they have enough snippets to produce a real summary).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.place_id, p.name, p.category, p.neighborhood
            FROM places p
            WHERE EXISTS (SELECT 1 FROM reviews_raw r WHERE r.place_id = p.place_id)
            ORDER BY EXISTS (SELECT 1 FROM ai_summaries s WHERE s.place_id = p.place_id) DESC
            LIMIT %(n)s;
            """,
            {"n": sample_size},
        )
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def run_variant(chain, place: dict, snippets: list[dict], aspect_rows: list[dict]) -> tuple[str, dict]:
    response = chain.invoke(
        {
            "place_name": place["name"],
            "category": place["category"],
            "neighborhood": place["neighborhood"] or "unknown area",
            "aspect_facts": format_aspect_facts(aspect_rows),
            "snippets": format_snippets(snippets),
        }
    )
    return response.content.strip(), response.usage_metadata or {}


def estimate_cost(usage: dict) -> float:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return round((input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    db_url = os.environ["DATABASE_URL"]
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)
    chain_a = ChatPromptTemplate.from_template(RAG_SUMMARY_PROMPT) | llm
    chain_b = ChatPromptTemplate.from_template(RAG_SUMMARY_PROMPT_V2) | llm

    tally = {"a": 0, "b": 0, "tie": 0}

    with psycopg.connect(db_url, connect_timeout=15) as conn:
        register_vector(conn)
        places = load_sample_places(conn, args.sample_size)
        log.info(f"Running {EXPERIMENT_NAME} on {len(places)} places")

        with conn.cursor() as cur:
            for place in places:
                snippets = retrieve_snippets(conn, embed_model, place)
                aspect_rows = fetch_aspect_facts(conn, place["place_id"])

                text_a, usage_a = run_variant(chain_a, place, snippets, aspect_rows)
                text_b, usage_b = run_variant(chain_b, place, snippets, aspect_rows)

                score_a = score_summary(text_a, aspect_rows)
                score_b = score_summary(text_b, aspect_rows)
                winner = pick_winner(score_a, score_b)
                tally[winner] += 1

                for variant, text, usage, score in (
                    ("a", text_a, usage_a, score_a),
                    ("b", text_b, usage_b, score_b),
                ):
                    cur.execute(
                        """
                        INSERT INTO experiment_results
                            (experiment_name, place_id, variant, output_text,
                             model_used, score_json, winner, cost_usd_est)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            EXPERIMENT_NAME,
                            place["place_id"],
                            variant,
                            text,
                            DEFAULT_MODEL,
                            psycopg.types.json.Json(score),
                            winner,
                            estimate_cost(usage),
                        ),
                    )
                log.info(f"  {place['name']!r}: winner={winner} (a={score_a['composite']:.1f}, b={score_b['composite']:.1f})")
        conn.commit()

    log.info(f"Done. Tally over {len(places)} places: {tally}")
    return tally


if __name__ == "__main__":
    main()
