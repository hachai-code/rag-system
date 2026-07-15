-- Review queue for the data flywheel: real production queries sampled from Langfuse
-- traces (evals/flywheel/sample_traces.py). A human reviews pending rows; accepted questions
-- get promoted into evals/eval_set.jsonl with a verified ideal answer.
CREATE TABLE IF NOT EXISTS candidate_eval_questions (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trace_id    TEXT        NOT NULL UNIQUE,    -- Langfuse trace id; makes ingestion idempotent
    question    TEXT        NOT NULL,           -- trace input (the user's real query)
    answer      TEXT,                           -- trace output (what the system said, unverified)
    retrieved   JSONB,                          -- [{id, title, distance}] from trace metadata
    trace_ts    TIMESTAMPTZ,                    -- when the production request happened
    status      TEXT        NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
