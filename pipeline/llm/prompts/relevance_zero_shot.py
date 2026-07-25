"""Phase 4 — zero-shot relevance classification.

Decides whether a piece of scraped text (Reddit post/comment, or any future
source) is worth spending further pipeline effort on. Runs only on text that
already survived the free keyword/location prefilter — this is the first
LLM-cost step, so it stays cheap and terse on purpose.
"""

RELEVANCE_PROMPT = """You are screening short pieces of text for a Copenhagen place-discovery app.

Decide if the text contains a genuine first-hand opinion, tip, or experience
about a specific restaurant, cafe, hotel, landmark, or similar place in
Copenhagen. It does not need to name the place explicitly if the location is
otherwise clear from context.

Answer "no" if the text is: asking for recommendations rather than giving one,
a general complaint about a booking platform or transport company, unrelated
to a specific place, or too vague to attach to any location.

Return strictly one JSON object, no markdown, no commentary:
{{"relevant": "yes" or "no", "reason": "<one short phrase>"}}

Text:
\"\"\"{text}\"\"\"
"""
