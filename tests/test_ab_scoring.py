import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "experiments"))
from ab_scoring import pick_winner, score_summary

ASPECT_ROWS = [
    {"aspect_category": "food", "avg_score": 4.2, "num_mentions": 3},
    {"aspect_category": "service", "avg_score": 3.5, "num_mentions": 2},
]


def test_high_aspect_coverage_scores_well():
    summary = "Great food and friendly service, a solid pick near Norrebro."
    result = score_summary(summary, ASPECT_ROWS)
    assert result["aspect_mentions"] == 2
    assert result["aspect_coverage"] == 1.0
    assert result["rule_violations"] == []


def test_no_aspect_mentions_scores_lower_than_full_coverage():
    summary = "A pleasant spot worth visiting if you're in the neighborhood."
    result = score_summary(summary, ASPECT_ROWS)
    full_coverage = score_summary("Great food and great service.", ASPECT_ROWS)
    assert result["composite"] < full_coverage["composite"]


def test_banned_phrase_is_flagged():
    summary = "As an AI, I can say the food here is good."
    result = score_summary(summary, ASPECT_ROWS)
    assert any("banned phrase" in v for v in result["rule_violations"])


def test_too_many_sentences_is_flagged():
    summary = ". ".join(["Sentence"] * 6) + "."
    result = score_summary(summary, ASPECT_ROWS)
    assert any("sentence_count" in v for v in result["rule_violations"])


def test_empty_aspect_rows_gives_none_coverage_not_error():
    result = score_summary("A nice place.", [])
    assert result["aspect_coverage"] is None
    assert result["composite"] == 0.0


def test_pick_winner_higher_composite_wins():
    a = score_summary("Great food and great service, highly recommended overall.", ASPECT_ROWS)
    b = score_summary("A place that exists somewhere in the city.", ASPECT_ROWS)
    assert pick_winner(a, b) == "a"


def test_pick_winner_tie():
    a = score_summary("Great food and service.", ASPECT_ROWS)
    b = score_summary("Great food and service.", ASPECT_ROWS)
    assert pick_winner(a, b) == "tie"
