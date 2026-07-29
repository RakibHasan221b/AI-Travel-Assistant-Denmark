"""Deterministic scoring for the RAG-summary prompt A/B test.

Deliberately not another LLM call to judge the LLM's output — same
regex/rules-before-LLM principle the salary agent uses (see the build plan's
Section 4 note): an LLM-as-judge would double the cost and add its own
variance to a test that's supposed to compare two *other* variants. Instead
this scores what's mechanically checkable against the prompt's own rules:
did the summary reference the rated aspects it was given, does it stay in
the requested length band, did it break the "don't mention you're an AI"
rule.
"""

import re

MIN_SENTENCES = 1
MAX_SENTENCES = 5
BANNED_PHRASES = ("as an ai", "this summary", "based on reviews", "based on the reviews")


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _aspect_mentions(text: str, aspect_rows: list[dict]) -> int:
    lowered = text.lower()
    return sum(1 for r in aspect_rows if r["aspect_category"].lower() in lowered)


def score_summary(summary_text: str, aspect_rows: list[dict]) -> dict:
    """Returns a metrics dict; higher `composite` is better. Pure function,
    no network/DB calls, safe to unit test directly.
    """
    text = summary_text.strip()
    sentence_count = _sentence_count(text)
    aspect_mentions = _aspect_mentions(text, aspect_rows)
    aspect_coverage = (aspect_mentions / len(aspect_rows)) if aspect_rows else None

    violations = []
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            violations.append(f"banned phrase: {phrase!r}")
    if not (MIN_SENTENCES <= sentence_count <= MAX_SENTENCES):
        violations.append(f"sentence_count {sentence_count} outside {MIN_SENTENCES}-{MAX_SENTENCES}")

    composite = (aspect_coverage or 0.0) * 10 - len(violations) * 5

    return {
        "word_count": len(text.split()),
        "sentence_count": sentence_count,
        "aspect_mentions": aspect_mentions,
        "aspect_coverage": aspect_coverage,
        "rule_violations": violations,
        "composite": composite,
    }


def pick_winner(score_a: dict, score_b: dict) -> str:
    """Returns 'a', 'b', or 'tie' by composite score."""
    if score_a["composite"] > score_b["composite"]:
        return "a"
    if score_b["composite"] > score_a["composite"]:
        return "b"
    return "tie"
