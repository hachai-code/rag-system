-- The two-agent stack's own tables, replacing the LangGraph checkpointer + PostgresStore.
-- Both are ADDITIVE: the old `store`/checkpoint tables persist until the Phase 8 cutover.

-- agent_threads: message-history persistence for the deep agent's HITL pause/resume.
-- pydantic-ai 0.8.1 has no Postgres durable execution (only Temporal), so a paused run
-- serializes its ModelMessage history here (ModelMessagesTypeAdapter dump) keyed by
-- thread_id; POST /ask/agent/resume loads it, feeds the approval tool results back, and
-- continues. One row per paused thread; overwritten on re-pause, deleted when the run ends.
CREATE TABLE IF NOT EXISTS agent_threads (
    thread_id  TEXT PRIMARY KEY,
    question   TEXT        NOT NULL,
    messages   JSONB       NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- qa_memory: the deep agent's Q&A semantic cache (replaces the LangGraph PostgresStore
-- Q&A store). The question is embedded with Voyage (1024 dims, input_type=document); a
-- new run matches past questions by cosine similarity. `value` holds the answer + sources
-- for reuse and the /qa views.
CREATE TABLE IF NOT EXISTS qa_memory (
    key        TEXT        PRIMARY KEY,
    question   TEXT        NOT NULL,
    value      JSONB       NOT NULL,
    embedding  VECTOR(1024) NOT NULL,       -- Voyage models (1024 dims)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ANN index on the question vectors, mirroring chunks_embedding_idx / chunk_questions.
CREATE INDEX IF NOT EXISTS qa_memory_embedding_idx
    ON qa_memory USING hnsw (embedding vector_cosine_ops);
