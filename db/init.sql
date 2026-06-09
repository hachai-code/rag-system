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
    metadata     JSONB  NOT NULL DEFAULT '{}',
    UNIQUE (document_id, chunk_index)
);

-- Approximate-nearest-neighbour index for cosine distance (the <=> operator).
-- Voyage embeddings are unit-normalized, so cosine is the natural metric.
-- At this corpus size pgvector would scan exactly anyway; the index is here so
-- the schema is complete and stays fast as the corpus grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
