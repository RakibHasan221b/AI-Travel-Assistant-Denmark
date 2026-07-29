-- AI Denmark Explorer — Phase 1 serving-layer schema (Copenhagen pilot)
-- See docs/architecture.md and the build plan for the rationale behind each table.

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector — semantic search (Phase 6), populated later, declared now to avoid a migration

CREATE TABLE places (
    place_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    osm_id          text UNIQUE NOT NULL,
    name            text NOT NULL,
    category        text NOT NULL,          -- restaurant | cafe | hotel | landmark | bar ...
    subcategory     text,
    lat             double precision NOT NULL,
    lon             double precision NOT NULL,
    address         text,
    neighborhood    text,
    opening_hours   text,
    osm_tags        jsonb NOT NULL DEFAULT '{}',
    price_level     smallint,
    embedding       vector(384),            -- sentence-transformers all-MiniLM-L6-v2, filled in Phase 6
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_places_category ON places (category);
CREATE INDEX idx_places_neighborhood ON places (neighborhood);
CREATE INDEX idx_places_osm_tags ON places USING gin (osm_tags);
CREATE INDEX idx_places_embedding ON places USING hnsw (embedding vector_cosine_ops);

-- All raw text, source-tagged: seed reviews, Wikivoyage descriptions, Reddit posts/comments
CREATE TABLE reviews_raw (
    review_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    place_id        uuid REFERENCES places (place_id),
    source_type     text NOT NULL CHECK (source_type IN ('seed', 'wikivoyage', 'reddit_post', 'reddit_comment')),
    source_id       text,
    source_url      text,
    author_hash     text,                   -- hashed, never raw username
    text_content    text NOT NULL,
    posted_at       timestamptz,
    run_id          text,
    raw_payload     jsonb,
    embedding       vector(384),            -- filled in Phase 6, used for RAG retrieval in Phase 8
    ingested_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_reviews_raw_place_id ON reviews_raw (place_id);
CREATE INDEX idx_reviews_raw_source_type ON reviews_raw (source_type);
CREATE INDEX idx_reviews_raw_run_id ON reviews_raw (run_id);
CREATE INDEX idx_reviews_raw_embedding ON reviews_raw USING hnsw (embedding vector_cosine_ops);

-- Fuzzy/exact linking of unlinked review text (mainly Reddit) to a place
CREATE TABLE place_mentions (
    mention_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id       uuid NOT NULL REFERENCES reviews_raw (review_id),
    place_id        uuid NOT NULL REFERENCES places (place_id),
    match_method    text NOT NULL CHECK (match_method IN ('exact_name', 'fuzzy', 'manual')),
    confidence      real NOT NULL DEFAULT 1.0
);

CREATE INDEX idx_place_mentions_place_id ON place_mentions (place_id);

-- LLM- and ML-derived aspect sentiment (the core signal table)
CREATE TABLE aspect_sentiment (
    sentiment_id    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id       uuid NOT NULL REFERENCES reviews_raw (review_id),
    place_id        uuid NOT NULL REFERENCES places (place_id),
    aspect_category text NOT NULL,          -- food | service | ambiance | value | location | overall
    sentiment_score smallint NOT NULL CHECK (sentiment_score BETWEEN 1 AND 5),
    confidence      real,
    model_used      text NOT NULL,          -- e.g. 'groq-llama-3.3-70b' | 'distilled-clf-v1' | 'seed-manual'
    run_id          text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_aspect_sentiment_place_id ON aspect_sentiment (place_id);

-- Fast-read rollup consumed by the app
CREATE TABLE aggregated_sentiment (
    place_id        uuid NOT NULL REFERENCES places (place_id),
    aspect_category text NOT NULL,
    avg_score       real NOT NULL,
    num_mentions    int NOT NULL DEFAULT 0,
    last_updated    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (place_id, aspect_category)
);

-- RAG-grounded AI summaries
CREATE TABLE ai_summaries (
    summary_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    place_id        uuid NOT NULL REFERENCES places (place_id),
    summary_text    text NOT NULL,
    model_used      text NOT NULL,
    prompt_version  text,
    sources         jsonb NOT NULL DEFAULT '[]',
    cost_usd_est    numeric(10, 6),
    generated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_ai_summaries_place_id ON ai_summaries (place_id);

-- Every ML-vs-LLM decision, incl. the distillation loop's fallback flag
CREATE TABLE ml_predictions (
    prediction_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    place_id          uuid REFERENCES places (place_id),
    review_id         uuid REFERENCES reviews_raw (review_id),
    target            text NOT NULL,        -- 'relevance' | 'quality_score' | 'aspect_sentiment'
    predicted_value    real NOT NULL,
    model_name        text NOT NULL,
    model_version     text,
    confidence        real,
    used_llm_fallback boolean NOT NULL DEFAULT false,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_ml_predictions_target ON ml_predictions (target);

-- Weather cache (Open-Meteo)
CREATE TABLE weather_daily (
    date            date PRIMARY KEY,
    temp_max_c      real,
    temp_min_c      real,
    precip_mm       real,
    wind_kph        real,
    condition_code  int,
    fetched_at      timestamptz NOT NULL DEFAULT now()
);

-- Time-series forecast output
CREATE TABLE visit_time_forecast (
    place_id                 uuid NOT NULL REFERENCES places (place_id),
    forecast_date            date NOT NULL,
    predicted_interest_score real NOT NULL,
    model_name               text NOT NULL,
    generated_at              timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (place_id, forecast_date, model_name)
);

-- Unsupervised clustering output
CREATE TABLE clusters (
    cluster_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    label           text,
    description     text,
    model_params    jsonb,
    generated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE place_clusters (
    place_id             uuid NOT NULL REFERENCES places (place_id),
    cluster_id           uuid NOT NULL REFERENCES clusters (cluster_id),
    distance_to_centroid real,
    PRIMARY KEY (place_id, cluster_id)
);

-- A/B test results for LLM prompt variants (e.g. Phase 8's RAG summary
-- prompt v1 vs v2) — both variants' output land here so a winner can be
-- picked by a deterministic score (pipeline/experiments/ab_scoring.py)
-- instead of eyeballing outputs.
CREATE TABLE experiment_results (
    result_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name text NOT NULL,          -- e.g. 'rag_summary_prompt_v1_vs_v2'
    place_id        uuid REFERENCES places (place_id),
    variant         text NOT NULL CHECK (variant IN ('a', 'b')),
    output_text     text NOT NULL,
    model_used      text NOT NULL,
    score_json      jsonb NOT NULL,          -- ab_scoring.score_summary() output
    winner          text CHECK (winner IN ('a', 'b', 'tie')),  -- same for both rows of a pair
    cost_usd_est    numeric(10, 6),
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_experiment_results_name ON experiment_results (experiment_name);
CREATE INDEX idx_experiment_results_place_id ON experiment_results (place_id);

-- Idempotent, resumable pipeline run tracking
CREATE TABLE pipeline_runs (
    run_id            text PRIMARY KEY,     -- e.g. cph-20260725-001
    stage             text NOT NULL,
    started_at        timestamptz NOT NULL DEFAULT now(),
    completed_at      timestamptz,
    status            text NOT NULL DEFAULT 'running',  -- running | completed | failed
    records_processed int,
    notes             text
);
