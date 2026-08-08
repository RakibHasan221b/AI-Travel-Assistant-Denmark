"""Deterministic reranking layer for /explore, applied after pgvector
semantic retrieval and before the final results are returned. Pure
functions, no DB/network access, so they're directly unit-testable.

Real problem this exists to fix: raw pgvector ORDER BY embedding <=> qvec
treats every result in the requested LIMIT as equally "found," even when
the tail is only weakly related. A real query for "little mermaid"
returned "Den lille Havfrue" (similarity 0.47) correctly at #1, but also
padded the list with completely unrelated hotels/restaurants/statues down
around 0.20-0.30 similarity, just because they were the next-nearest
vectors — semantic similarity was being treated as relevance on its own,
which it isn't strong enough to be by itself.

The fix is two real signals combined, not a bigger/smaller semantic
threshold alone:
  1. name_match_score - deterministic lexical matching (exact/substring/
     token-overlap) against the place's own name, so an obvious entity
     search ("Restaurant Klubben") can heavily outweigh other places that
     merely have nearby embeddings (e.g. "Madklubben", still restaurants,
     still moderately similar semantically, but not the same entity).
  2. category_intent_score - a small, explicit keyword->category map for
     common intent words ("coffee" -> cafe, "museum" -> landmark), so a
     category-shaped query favors matching categories without excluding
     everything else outright.

Both empirically tested against the real database (not guessed) for five
real queries spanning both an entity search ("little mermaid",
"Restaurant Klubben") and category searches ("coffee", "sushi",
"museum") before this exact threshold was chosen - see the commit message
for the real before/after result lists.
"""

import re
import unicodedata

# Deliberately small and literal - covers the category-shaped queries this
# was actually tested against, not an attempt at exhaustive intent
# classification. A query word not in this map simply gets no category
# boost, falling back to semantic + name matching alone, which is the
# safe default already proven to work for entity searches.
_CATEGORY_KEYWORDS = {
    "coffee": "cafe", "cafe": "cafe", "café": "cafe",
    "museum": "landmark", "gallery": "landmark", "landmark": "landmark",
    "restaurant": "restaurant", "dinner": "restaurant", "lunch": "restaurant",
    "hotel": "hotel", "stay": "hotel",
    "bar": "bar", "pub": "bar", "drinks": "bar",
}

# A candidate needs at least this much combined score to be considered
# relevant at all - real garbage-tail entries for "little mermaid" (random
# statues, hotels) topped out around 0.30 combined; every real match in
# the five test queries scored well above this. Also needs to be within
# this margin of the single best match - real entity searches (e.g.
# "Restaurant Klubben", combined 1.38 for the exact place) have semantic
# neighbors sitting at 0.6-0.9 that would otherwise still clear the floor
# above on raw similarity alone; the relative gap is what actually
# separates "the place you meant" from "other places like it."
RELEVANCE_FLOOR = 0.40
RELEVANCE_GAP = 0.30

# Wider than the number of results actually shown - the real
# "Den lille Havfrue" example needs the true match to survive as a
# candidate even when it isn't in the raw embedding-only top few, so this
# has to be generous enough that name/category boosts have a real
# candidate to promote from the second half of the pool, not just the
# already-good top of it.
DEFAULT_POOL_SIZE = 40


def normalize_text(s: str) -> str:
    """Casefold, strip accents/punctuation, collapse whitespace - shared
    normalization so "café" and "Cafe" (or "Klubben" and "klubben,") are
    treated as the same token for matching purposes."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return set(normalize_text(s).split())


def name_match_score(query: str, name: str, subcategory: str | None = None) -> float:
    """0.0-1.0. Exact normalized match (1.0) beats substring containment
    in either direction (0.7) beats partial token overlap (up to 0.5,
    scaled by what fraction of the QUERY's own tokens actually appear in
    the name/subcategory) beats no lexical relationship at all (0.0).
    Deliberately does not attempt cross-language/fuzzy matching (e.g.
    English "mermaid" vs Danish "havfrue") - that case is real and known,
    but is left to semantic similarity, which already handles it
    correctly on its own (verified: "Den lille Havfrue" already ranks
    first by raw similarity for "little mermaid")."""
    nq, nn = normalize_text(query), normalize_text(name)
    if not nq or not nn:
        return 0.0
    if nq == nn:
        return 1.0
    if nq in nn or nn in nq:
        return 0.7
    qtok = _tokens(query)
    ntok = _tokens(name) | _tokens(subcategory or "")
    if not qtok or not ntok:
        return 0.0
    overlap = len(qtok & ntok) / len(qtok)
    return overlap * 0.5


def category_intent_score(query: str, category: str) -> float:
    """+0.15 if any token in the query names a known category and the
    candidate's own category matches it, else 0.0 - a boost, not a
    filter, so a place outside the inferred category can still surface
    via strong semantic/name signals rather than being excluded outright."""
    for tok in _tokens(query):
        wanted = _CATEGORY_KEYWORDS.get(tok)
        if wanted and category == wanted:
            return 0.15
    return 0.0


def rank_explore_candidates(
    query: str,
    candidates: list[dict],
    final_limit: int = 5,
) -> list[dict]:
    """candidates: dicts with at least name/category/subcategory/similarity
    (similarity = raw pgvector cosine similarity, 0-1), already ordered by
    the caller however it likes - this doesn't assume an incoming order.
    Returns the same dicts (unmodified, `similarity` untouched - it stays
    the real semantic score for display, never overwritten by the
    combined ranking score, which is internal to this function only),
    reordered and filtered, capped at final_limit. May return fewer than
    final_limit - even zero - rather than padding with weak matches; see
    this module's docstring for why that's the intended behavior, not a
    bug to guard against upstream."""
    scored = []
    for c in candidates:
        ns = name_match_score(query, c["name"], c.get("subcategory"))
        cs = category_intent_score(query, c["category"])
        combined = c["similarity"] + 0.45 * ns + cs
        scored.append((combined, c))

    scored.sort(key=lambda t: -t[0])
    top = scored[0][0] if scored else 0.0
    kept = [c for combined, c in scored if combined >= RELEVANCE_FLOOR and combined >= top - RELEVANCE_GAP]

    # Real duplicate-branch problem found during testing: "coffee" and
    # "sushi" both had one popular real chain (e.g. "Original Coffee",
    # "Sticks'n'Sushi") occupying every one of the top 5 slots by itself,
    # each real location scored close enough to survive the filter above.
    # Keeping only the best-scoring instance per distinct name turns that
    # into genuine variety instead - the same UX call a normal place-
    # search product makes, not a data dedup (nothing is deleted, this is
    # display-only).
    seen_names: set[str] = set()
    deduped = []
    for c in kept:
        key = normalize_text(c["name"])
        if key in seen_names:
            continue
        seen_names.add(key)
        deduped.append(c)

    return deduped[:final_limit]
