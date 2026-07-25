"""Phase 7 — few-shot cluster naming.

Runs once per cluster (not once per place), on the handful of places closest
to each KMeans/DBSCAN centroid. Keeps this LLM cost trivial regardless of
how many places are in the dataset.
"""

CLUSTER_NAMING_PROMPT = """You name groups of similar places in Copenhagen based on a few examples from each group.

Examples:

Group: Places tagged cafe, high ambiance scores, near water, mentioned for "quiet" and "work"
Output: {{"label": "Waterfront work cafes", "description": "Calm cafes by the water, good for working or reading."}}

Group: Places tagged restaurant, high value scores, casual, frequently mentioned for lunch
Output: {{"label": "Budget-friendly lunch spots", "description": "Casual restaurants known for good value at lunchtime."}}

Now name this group based on its most representative places:

{exemplar_places}

Return strictly one JSON object, no markdown, no commentary:
{{"label": "<2-4 words>", "description": "<one sentence>"}}
"""
