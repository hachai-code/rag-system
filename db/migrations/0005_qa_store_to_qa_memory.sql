-- Carry the Q&A cache from the LangGraph PostgresStore (`store` + `store_vectors`) over to
-- `qa_memory`, so the Phase 8 cutover can drop the old tables without losing the cache.
--
-- Lossless copy, no re-embedding: the old value jsonb already has exactly the shape
-- save_qa_record writes (question/answer/corpus_sources/web_urls/research_files), the
-- question vectors are the same Voyage 1024-dim space, and the keys are the same uuid4().hex
-- form. So value and embedding transfer verbatim and created_at is preserved.
--
-- Guarded on `store` existing: a DB provisioned fresh (or already cut over) has no
-- PostgresStore tables and must skip this rather than fail. ON CONFLICT keeps re-runs no-ops.
DO $$
BEGIN
    IF to_regclass('public.store') IS NOT NULL
       AND to_regclass('public.store_vectors') IS NOT NULL THEN
        INSERT INTO qa_memory (key, question, value, embedding, created_at)
        SELECT s.key, s.value ->> 'question', s.value, sv.embedding, s.created_at
        FROM store s
        JOIN store_vectors sv
          ON sv.prefix = s.prefix
         AND sv.key = s.key
         AND sv.field_name = 'question'
        WHERE s.prefix = 'qa'
          AND s.value ? 'question'
        ON CONFLICT (key) DO NOTHING;
    END IF;
END $$;
