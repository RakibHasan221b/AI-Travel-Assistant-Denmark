"""Offline, pure-function tests for api/ranking.py — no DB/network access,
matching this project's existing test conventions. Real problem this
guards: raw pgvector ORDER BY treated every result in the requested LIMIT
as equally relevant — a real query for "little mermaid" correctly ranked
"Den lille Havfrue" first (similarity 0.47) but padded the rest of the
list with unrelated hotels/restaurants/statues down around 0.20-0.30,
just because they were the next-nearest embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.ranking import (
    RELEVANCE_FLOOR,
    RELEVANCE_GAP,
    category_intent_score,
    name_match_score,
    normalize_text,
    rank_explore_candidates,
)


def test_normalize_text_strips_accents_case_and_punctuation():
    assert normalize_text("Café!") == normalize_text("cafe")
    assert normalize_text("Restaurant Klubben,") == "restaurant klubben"


def test_name_match_score_exact_normalized_match_is_perfect():
    assert name_match_score("Restaurant Klubben", "Restaurant Klubben", None) == 1.0
    assert name_match_score("restaurant klubben", "Restaurant Klubben", None) == 1.0


def test_name_match_score_substring_containment_scores_high_but_not_perfect():
    assert name_match_score("Klubben", "Restaurant Klubben", None) == 0.7


def test_name_match_score_no_lexical_relationship_scores_zero():
    # The real, known case this deliberately does NOT solve — "little
    # mermaid" vs "Den lille Havfrue" shares no tokens (English vs
    # Danish); semantic similarity already handles that case correctly on
    # its own (verified separately against the real DB).
    assert name_match_score("little mermaid", "Den lille Havfrue", None) == 0.0


def test_category_intent_score_matches_a_known_keyword():
    assert category_intent_score("coffee near me", "cafe") == 0.15
    assert category_intent_score("coffee near me", "restaurant") == 0.0
    assert category_intent_score("nothing relevant here", "cafe") == 0.0


def _candidate(name, category, similarity, subcategory=None):
    return {"name": name, "category": category, "subcategory": subcategory, "similarity": similarity}


def test_rank_explore_candidates_keeps_only_the_dominant_entity_match():
    # Real data this mirrors: "Restaurant Klubben" (combined ~1.38 after
    # the exact-name boost) vs. semantically-similar-but-different
    # restaurants sitting at 0.6-0.73 raw similarity with zero name boost
    # — the relative gap is what actually separates them.
    candidates = [
        _candidate("Restaurant Klubben", "restaurant", 0.780),
        _candidate("Madklubben Østerbro", "restaurant", 0.730),
        _candidate("Restaurant Vie", "restaurant", 0.668),
    ]
    result = rank_explore_candidates("Restaurant Klubben", candidates, final_limit=5)
    assert [r["name"] for r in result] == ["Restaurant Klubben"]


def test_rank_explore_candidates_excludes_the_weak_tail():
    # Mirrors the real "little mermaid" garbage-tail case: the true match
    # survives, unrelated low-similarity results (hotels, other statues)
    # don't just because they were in the raw top-N.
    candidates = [
        _candidate("Den lille Havfrue", "landmark", 0.465),
        _candidate("Valkyrie", "landmark", 0.304),
        _candidate("Tiffany", "hotel", 0.236),
        _candidate("Islands Brygge Wok", "restaurant", 0.251),
    ]
    result = rank_explore_candidates("little mermaid", candidates, final_limit=5)
    assert [r["name"] for r in result] == ["Den lille Havfrue"]


def test_rank_explore_candidates_keeps_real_category_matches_up_to_the_limit():
    candidates = [
        _candidate("Roast Coffee", "cafe", 0.645),
        _candidate("Original Coffee", "cafe", 0.634),
        _candidate("Switch Coffee", "cafe", 0.610),
        _candidate("Coffee Coach", "cafe", 0.595),
        _candidate("The Coffee Factory", "cafe", 0.592),
        _candidate("Coffee Industry", "cafe", 0.591),
    ]
    result = rank_explore_candidates("coffee", candidates, final_limit=5)
    assert len(result) == 5
    assert all(r["category"] == "cafe" for r in result)


def test_rank_explore_candidates_dedupes_by_normalized_name():
    # Real problem found during testing: one popular real chain (multiple
    # real branches, same name) filled every slot in the top 5 by itself.
    candidates = [
        _candidate("Original Coffee", "cafe", 0.634),
        _candidate("Original Coffee", "cafe", 0.620),
        _candidate("Original Coffee", "cafe", 0.557),
        _candidate("Switch Coffee", "cafe", 0.610),
    ]
    result = rank_explore_candidates("coffee", candidates, final_limit=5)
    names = [r["name"] for r in result]
    assert names.count("Original Coffee") == 1
    assert "Switch Coffee" in names


def test_rank_explore_candidates_returns_fewer_than_the_limit_when_only_one_matches():
    # Explicit requirement: do not pad the page with weak matches just to
    # reach the requested count.
    candidates = [
        _candidate("Den lille Havfrue", "landmark", 0.465),
        _candidate("Valkyrie", "landmark", 0.304),
    ]
    result = rank_explore_candidates("little mermaid", candidates, final_limit=5)
    assert len(result) == 1


def test_rank_explore_candidates_returns_empty_when_nothing_clears_the_floor():
    candidates = [_candidate("Random Place", "restaurant", 0.15)]
    result = rank_explore_candidates("completely unrelated gibberish query", candidates, final_limit=5)
    assert result == []


def test_rank_explore_candidates_never_overwrites_the_real_similarity_field():
    # The combined score is internal to ranking only — the returned dict's
    # `similarity` must stay the real semantic score for display, never
    # the boosted ranking score.
    candidates = [_candidate("Restaurant Klubben", "restaurant", 0.780)]
    result = rank_explore_candidates("Restaurant Klubben", candidates, final_limit=5)
    assert result[0]["similarity"] == 0.780


def test_relevance_constants_are_the_ones_documented():
    # Guards against a silent tuning drift — these exact values were
    # chosen against real query score distributions (see the module
    # docstring / commit message), not arbitrary.
    assert RELEVANCE_FLOOR == 0.40
    assert RELEVANCE_GAP == 0.30
