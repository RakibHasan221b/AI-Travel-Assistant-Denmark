"""Shared Groq client — retry/backoff, strict JSON-mode responses.

First real LLM usage in the project. Groq's free tier is rate-limited
(~30 req/min, ~1,000 req/day baseline), not billed, so failures here are
almost always transient rate limits, not cost or auth problems - the retry
logic reflects that.
"""

import json
import logging
import os
import time

import requests

log = logging.getLogger("groq_client")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqError(Exception):
    pass


def call_groq(prompt: str, model: str = DEFAULT_MODEL, max_retries: int = 4) -> dict:
    api_key = os.environ["GROQ_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            wait = min(2**attempt, 30)
            log.warning(f"network error (attempt {attempt}/{max_retries}): {e}, waiting {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", min(10 * attempt, 60)))
            log.warning(f"rate limited (attempt {attempt}/{max_retries}), waiting {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code in (401, 403):
            raise GroqError(f"auth error: {resp.status_code} {resp.text}")

        if resp.status_code >= 500:
            wait = min(2**attempt, 30)
            log.warning(f"server error {resp.status_code} (attempt {attempt}/{max_retries}), waiting {wait}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start : end + 1])
            raise GroqError(f"could not parse JSON from response: {content[:200]}")

    raise GroqError(f"exhausted {max_retries} attempts")
