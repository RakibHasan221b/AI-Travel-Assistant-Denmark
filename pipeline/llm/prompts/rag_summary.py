"""Phase 8 — RAG-grounded place summary.

Reserved for GPT-4o. Retrieved snippets come from the FAISS index (Phase 6).
The prompt forces citation to specific sources and explicitly forbids
inventing detail beyond what's retrieved — the whole point of RAG here is to
stop the LLM from hallucinating a place it doesn't actually have data on.
"""

RAG_SUMMARY_PROMPT = """Write a short, honest summary of a place in Copenhagen using ONLY the
retrieved snippets below. Do not add facts, prices, or opinions that aren't
in the snippets.

Rules:
- 2-4 sentences.
- If the snippets disagree with each other, say so briefly instead of picking one side.
- If the snippets are thin (fewer than 3), say the picture is limited rather than
  writing confidently anyway.
- Do not mention that you are an AI or that this is a summary of reviews — just write it.

Place: {place_name} ({category}, {neighborhood})

Retrieved snippets:
{snippets}

Summary:
"""
