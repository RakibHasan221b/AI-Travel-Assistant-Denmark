import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from web_enrichment import MAX_RESULTS_PER_PLACE, filter_results

GOOD_RESULT = {
    "title": "The Little Mermaid (statue)",
    "link": "https://en.wikipedia.org/wiki/The_Little_Mermaid_(statue)",
    "snippet": "The sculpture is displayed on a rock by the waterside at the Langelinie promenade in Copenhagen, Denmark.",
}


def test_good_result_is_kept():
    assert filter_results([GOOD_RESULT]) == [GOOD_RESULT]


def test_low_signal_domain_is_dropped():
    result = {**GOOD_RESULT, "link": "https://www.facebook.com/groups/example/posts/123"}
    assert filter_results([result]) == []


def test_thin_snippet_is_dropped():
    result = {**GOOD_RESULT, "snippet": "Nice place."}
    assert filter_results([result]) == []


def test_missing_snippet_is_dropped_not_crashed():
    result = {"title": "No snippet here", "link": "https://example.com"}
    assert filter_results([result]) == []


def test_caps_at_max_results():
    results = [
        {**GOOD_RESULT, "link": f"https://en.wikipedia.org/wiki/Place_{i}"}
        for i in range(10)
    ]
    assert len(filter_results(results)) == MAX_RESULTS_PER_PLACE
