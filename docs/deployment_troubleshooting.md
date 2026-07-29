# Deployment troubleshooting log (Phase 12, Render API service)

A real record of what broke deploying `ai-denmark-explorer-api` to Render
and how each was actually root-caused — kept for interview prep ("tell me
about a time you debugged a production issue") and so the same mistakes
don't get repeated. Each entry is a genuinely separate failure, not restarts
of the same one — the deploy got further each time.

## 1. Build failed: `tiktoken` wheel build failed (Rust/Cargo, read-only filesystem)

**Symptom:**
```
error: failed to create directory `/usr/local/cargo/registry/cache/...`
Caused by: Read-only file system (os error 30)
ERROR: Failed building wheel for tiktoken
```

**Root cause:** `render.yaml` never pinned a Python version, so Render
defaulted to Python 3.14 — much newer than this project had ever run on
(CI and local dev both use 3.11, per `pyproject.toml`'s
`requires-python = ">=3.11"`). `tiktoken` (a `litellm` dependency, pulled in
transitively) has no prebuilt wheel for 3.14, so pip fell back to compiling
it from Rust source. That build needs to write to a Cargo cache directory,
and Render's build sandbox mounts that path read-only — the compile can
never succeed there, on any package, regardless of the actual Rust code.

**Fix:** Pinned `PYTHON_VERSION: "3.11.9"` as an env var in `render.yaml`,
matching CI and local dev. No code change — purely an environment mismatch.

**How it was found:** Render's deploy event only said "Exited with status 1
while building your code" — not useful alone. The actual reason was in the
service's **Build Logs** tab, not the event summary. Lesson: always go to
the full build log, not just the event/notification text.

---

## 2. Build succeeded, but startup crashed: `ModuleNotFoundError: No module named 'pgvector'`

**Symptom:** Build finished clean, but `uvicorn` crashed immediately on
import with a missing-module error for `pgvector`.

**Root cause:** `render.yaml`'s build command is `pip install -e ".[agent]"`
— only the `agent` extra, not every extra. But `agent/tools.py`'s
`search_places` tool does real pgvector semantic search, so it needs the
`pgvector` package. That dependency was declared under the `embeddings`
extra, not `agent`. This never surfaced locally or in CI because both
always install *every* extra together (`.[dev,embeddings,rag,agent]`) —
Render's narrower, single-extra install was the first environment that ever
exposed the real gap between "what `agent/tools.py` imports" and "what the
`agent` extra declares."

**Fix:** Declared the real dependency directly in the `agent` extra.

**Lesson:** An extras/dependency-group system is only correct if each group
is tested in isolation at least once — installing "everything" everywhere
else had been silently masking a real gap since Phase 11 was first built.

---

## 3. Build + startup succeeded, but the process got killed: `Exited with status 137`

**Symptom:** Build succeeded, `uvicorn` started, then after ~2 minutes of
"No open ports detected" the process died with status 137 — never actually
bound to a port.

**Root cause:** Status 137 = 128 + 9 = killed by `SIGKILL`, the classic
signature of Linux's out-of-memory killer. Render's free tier gives 512MB
RAM. The dependency stack pulled in by this service — `crewai`, `torch`,
`transformers`, `chromadb`, `lancedb`, `onnxruntime` — is heavy; something
in that chain exceeded 512MB before the app ever got to open its port.

This was the exact risk `render.yaml`'s own comment had flagged from the
start as "not yet verified live" — now confirmed true, but the *specific*
cause wasn't obvious (is `crewai` itself just this heavy, meaning nothing
short of a paid tier fixes it, or is something in *our* code responsible?).

**Diagnosis (before touching any code or spending another deploy cycle):**
tested locally, in the isolated `agent` venv:
```python
import sys
import crewai
print('torch' in sys.modules)        # False
print('transformers' in sys.modules) # False
print('chromadb' in sys.modules)     # True (lightweight at import time)
```
`crewai` alone does **not** pull in torch/transformers. Then:
```python
from sentence_transformers import SentenceTransformer
print('torch' in sys.modules)  # True
```
It was `agent/tools.py`'s own `from sentence_transformers import
SentenceTransformer` (used to embed the search query for pgvector search)
that dragged in torch + transformers — not `crewai` itself. Confirmed via
`pip show crewai chromadb litellm fastapi`: none of their `Requires:` lists
include torch or transformers at all.

**Fix:** Swapped `sentence-transformers` for **fastembed** (ONNX Runtime —
already installed anyway, as a transitive dependency of `chromadb`) in the
live API path only. fastembed ships the same `all-MiniLM-L6-v2` weights as
an ONNX export, so the vectors it produces are compatible with what
Phase 6's batch ingestion script (which keeps using sentence-transformers —
it runs locally, not on Render, so its footprint was never the problem)
already wrote to the DB. Verified before switching, not assumed:

```
cosine similarity between sentence-transformers and fastembed output: 1.0000
```

Then re-verified `search_places` against the real DB with the new
embedding — query "cozy quiet cafe good for working" returned a place
literally named "Cozy" as the top hit.

**Lesson:** "Framework X is heavy" is often actually "one specific
dependency of X is heavy, and it's not even X's own requirement." Measuring
*which import* actually pulls in the weight (`'torch' in sys.modules`) took
two minutes and turned a "pay for a bigger server" problem into a genuinely
free fix — worth doing before assuming a paid tier is the only way out.

---

## Net result

Three separate, real bugs — a Python-version/build-sandbox mismatch, an
extras-declaration gap, and a memory-footprint problem — fixed across three
commits, entirely on Render's free tier, no functionality removed. All
verified against the live database and a live deployment, not just locally.
