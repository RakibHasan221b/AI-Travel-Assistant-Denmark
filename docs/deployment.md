# Deployment (Phase 12)

Two separate services, matching why Phase 11 split the crew out from the app
in the first place (see `docs/architecture.md`): the Streamlit app can't host
the CrewAI crew itself because Streamlit reruns the whole script on every
interaction, which fights the crew's need for stable process state.

Both steps below need your own GitHub/Render/Streamlit account login —
that's an OAuth/account-linking step only you can do, not something that can
be scripted from here.

## 1. API — Render

1. Go to [render.com](https://render.com), sign in with GitHub.
2. New → Blueprint → select this repo
   (`RakibHasan221b/AI-Travel-Assistant-Denmark`). Render reads `render.yaml`
   from the repo root automatically.
3. In the service's **Environment** tab, set:
   - `DATABASE_URL` — your Neon connection string
   - `GROQ_API_KEY` — your Groq key
4. Deploy. Once live, note the service URL (e.g.
   `https://ai-denmark-explorer-api.onrender.com`) — you'll need it for step 2.
5. Verify: `curl https://<your-render-url>/health` should return
   `{"status":"ok"}`.

**Known risk, not yet verified live:** the `agent` extra's dependency tree
(`crewai` → torch/transformers/chromadb/lancedb/onnxruntime) is heavy —
~1.8GB installed locally. Render's free tier has limited RAM/disk; this may
need a paid instance type, or just have a slow cold start on free. Watch the
build logs on first deploy — if it fails on memory during `pip install`,
that's the likely cause.

## 2. App — Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub.
2. New app → select this repo, branch `main`, main file path `app/Home.py`.
   Streamlit Cloud reads `requirements.txt` from the repo root automatically
   (deliberately kept separate and lighter than `pyproject.toml`'s full
   extras — see the comment at the top of that file).
3. In **Advanced settings → Secrets**, paste (TOML format):
   ```toml
   DATABASE_URL = "postgresql://..."
   TRIP_PLANNER_API_URL = "https://<your-render-url-from-step-1>"
   ```
   `app/common.py` reads `st.secrets` first, falling back to `.env` locally
   — no code changes needed between local and deployed.
4. Deploy. First load will be slow (`sentence-transformers` downloading the
   embedding model) — subsequent loads are fast.

## Verifying it's actually live

- Home page: place/cluster/summary counts should match what's in the repo's
  own README status table (1,468 places, 18 clusters, 175 summaries).
- Explore page: search "cozy quiet cafe" and confirm real results with
  quality scores.
- Trip Planner: takes 1-2 minutes (three real agent LLM calls) — this is
  expected, not a bug. See `agent/crew.py`'s module docstring for the two
  real upstream issues already worked around here.
