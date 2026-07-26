-- Phase 8 cutover: drop the LangGraph checkpointer + PostgresStore tables.
--
-- Nothing references them any more: the deep agent's HITL history lives in agent_threads
-- and its Q&A cache in qa_memory (carried over by 0005). All LangGraph code was deleted in
-- Phase 3, so the checkpoint_* tables are orphaned state.
--
-- This is irreversible, so the Q&A cache is guarded rather than trusted: if any `qa` record
-- in `store` did not make it into `qa_memory` (e.g. 0005's join skipped a row with no
-- question vector), abort the whole migration instead of dropping the only copy.
DO $$
DECLARE
    uncarried int;
BEGIN
    IF to_regclass('public.store') IS NOT NULL THEN
        SELECT count(*) INTO uncarried
        FROM store s
        WHERE s.prefix = 'qa'
          AND NOT EXISTS (SELECT 1 FROM qa_memory m WHERE m.key = s.key);

        IF uncarried > 0 THEN
            RAISE EXCEPTION
                'Refusing to drop store: % qa record(s) are not in qa_memory. Migration 0005 did not carry them; fix that before cutting over.',
                uncarried;
        END IF;
    END IF;
END $$;

-- store_vectors is a child of store (store_vectors_prefix_key_fkey), so it goes first.
-- Dropping in dependency order rather than with CASCADE, which could reach objects
-- outside this list.
DROP TABLE IF EXISTS store_vectors;
DROP TABLE IF EXISTS store;
DROP TABLE IF EXISTS store_migrations;
DROP TABLE IF EXISTS vector_migrations;

-- LangGraph checkpointer: no FKs between these, order is free.
DROP TABLE IF EXISTS checkpoint_writes;
DROP TABLE IF EXISTS checkpoint_blobs;
DROP TABLE IF EXISTS checkpoints;
DROP TABLE IF EXISTS checkpoint_migrations;
