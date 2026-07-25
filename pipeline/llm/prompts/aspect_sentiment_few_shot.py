"""Phase 4 — few-shot aspect-based sentiment scoring.

Runs only on text that already passed the zero-shot relevance check. Scores
each mentioned aspect separately instead of one overall sentiment number,
matching the aspect_sentiment table (aspect_category, sentiment_score 1-5).
"""

ASPECT_SENTIMENT_PROMPT = """You score opinions about places in Copenhagen, aspect by aspect.

For each aspect actually discussed in the text, output one entry with:
- aspect: one of food, service, ambiance, value, location, overall
- sentiment_score: integer 1-5 (1 = very negative, 3 = neutral/mixed, 5 = very positive)

Only include an aspect if the text actually expresses an opinion about it.
Do not invent aspects that aren't discussed.

Examples:

Text: "Coffee at this place is great but you'll wait 20 minutes on weekends, and it's not cheap."
Output: {{"aspects": [{{"aspect": "food", "sentiment_score": 5}}, {{"aspect": "value", "sentiment_score": 2}}]}}

Text: "Nice quiet spot to work, staff were friendly and the wifi was fast."
Output: {{"aspects": [{{"aspect": "ambiance", "sentiment_score": 5}}, {{"aspect": "service", "sentiment_score": 4}}]}}

Text: "It was fine. Nothing special, nothing wrong with it."
Output: {{"aspects": [{{"aspect": "overall", "sentiment_score": 3}}]}}

Return strictly one JSON object, no markdown, no commentary.

Text:
\"\"\"{text}\"\"\"
"""
