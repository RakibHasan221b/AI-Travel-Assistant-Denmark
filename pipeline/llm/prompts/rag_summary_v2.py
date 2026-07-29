"""RAG summary prompt — variant B, for the A/B test in pipeline/experiments.

v1 (rag_summary.py) writes a free-flowing 2-4 sentence summary. v2 tests a
different structure on purpose: a one-line verdict first, then supporting
detail — the hypothesis being that a scannable verdict-first shape reads
better in the Explore page's card layout than a paragraph does. Same
grounding rules as v1 (retrieved snippets + rated aspects only, no invented
facts) so the comparison isolates structure/tone, not groundedness rules.
"""

RAG_SUMMARY_PROMPT_V2 = """Write a short, honest summary of a place in Copenhagen using ONLY the
retrieved snippets and rated aspects below. Do not add facts, prices, or
opinions that aren't in them.

Rules:
- Start with a single one-line verdict sentence (who it's good for / not for).
- Follow with 1-2 supporting sentences citing what backs the verdict up.
- If the snippets disagree with each other, say so briefly instead of picking one side.
- If the snippets are thin (fewer than 3), say the picture is limited rather than
  writing confidently anyway.
- Do not mention that you are an AI or that this is a summary of reviews — just write it.

Place: {place_name} ({category}, {neighborhood})

Rated aspects (computed averages, treat as given facts):
{aspect_facts}

Retrieved snippets:
{snippets}

Summary:
"""
