-- Runs automatically the first time the container initializes an empty data
-- volume (docker compose up on a fresh `rag-pgdata`). This is the single
-- source of truth for the schema, so a clean `down -v && up` rebuilds the DB.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source file from the corpus. Mirrors ingest.py's Document.
CREATE TABLE IF NOT EXISTS documents (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source      TEXT        NOT NULL UNIQUE,   -- path relative to corpus root
    title       TEXT        NOT NULL,
    section     TEXT,                          -- top-level folder, e.g. "Maia"
    doc_date    DATE,                          -- file modified date
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk we embed. Deleting a document removes its chunks.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT    NOT NULL,              -- position within the document
    content      TEXT   NOT NULL,
    embedding    VECTOR(1024),                 -- Voyage models (1024 dims)
    -- Lexical index of content, kept in sync automatically. Powers keyword
    -- (full-text) search alongside the vector column — the two complement each
    -- other: vectors catch paraphrase, this catches exact rare terms.
    content_tsv  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata     JSONB  NOT NULL DEFAULT '{}',
    UNIQUE (document_id, chunk_index)
);

-- Approximate-nearest-neighbour index for cosine distance (the <=> operator).
-- Voyage embeddings are unit-normalized, so cosine is the natural metric.
-- At this corpus size pgvector would scan exactly anyway; the index is here so
-- the schema is complete and stays fast as the corpus grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Inverted index for full-text search (the @@ operator / ts_rank).
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING gin (content_tsv);

-- One row per judge run (one invocation of evals/answer_system/judge.py). `config`
-- records the knobs that define the run (judge model, eval file, RAG settings) so a
-- result is reproducible from its run row + git_sha.
CREATE TABLE IF NOT EXISTS eval_runs (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    git_sha     TEXT        NOT NULL,           -- HEAD at run time; "-dirty" if uncommitted
    config      JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per (run, eval question). `scores`/`rationales` are keyed by axial-code
-- dimension (A-E) — only the dimensions that question was judged on. cost is the USD
-- of this question's judge calls; latency_ms is their wall time.
CREATE TABLE IF NOT EXISTS eval_results (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id       BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id  INT    NOT NULL,               -- id from rag_system_human_eval.jsonl
    question     TEXT   NOT NULL,
    answer       TEXT   NOT NULL,               -- the RAG answer that was judged
    scores       JSONB  NOT NULL,               -- {"A": true, "C": false}
    rationales   JSONB  NOT NULL,               -- {"A": "...", "C": "..."}
    cost         NUMERIC NOT NULL,              -- USD, judge calls for this question
    latency_ms   INT     NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, question_id)
);
