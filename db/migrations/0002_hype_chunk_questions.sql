-- HyPE (Hypothetical Prompt Embeddings): index-time hypothetical questions that a
-- chunk answers, embedded separately. Matching a real query against these question
-- vectors closes the query/document phrasing gap. This is ADDITIVE — chunks.content
-- and chunks.embedding stay the raw chunk; a match here maps back to its parent chunk.
CREATE TABLE IF NOT EXISTS chunk_questions (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_id   BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    question   TEXT   NOT NULL,
    embedding  VECTOR(1024)                    -- Voyage models (1024 dims)
);

-- ANN index on the question vectors, mirroring chunks_embedding_idx.
CREATE INDEX IF NOT EXISTS chunk_questions_embedding_idx
    ON chunk_questions USING hnsw (embedding vector_cosine_ops);
