"""Data flywheel: sample real production queries into the candidate_eval_questions queue.

Every /ask request already lands in Langfuse with the user's question (input), the
answer (output), and the retrieved chunk summaries (metadata) on the rag-ask span —
see rag/app.py. This job copies those spans into Postgres as review candidates: a
human grades the pending rows and promotes accepted questions into
evals/eval_set.jsonl, so the eval set keeps growing from real usage. Pure copy job,
no LLM calls.

Reads the observations API (not traces): since FastAPI auto-instrumentation, the
trace root is the HTTP span ("POST /ask"), so question/answer live on the rag-ask
observation, not at trace level.

Idempotent — trace_id is UNIQUE with ON CONFLICT DO NOTHING, so re-running (by hand
or cron) only adds spans it hasn't seen.

    uv run python -m evals.flywheel.sample_traces

Review queue:  SELECT id, trace_ts, question FROM candidate_eval_questions
               WHERE status = 'pending' ORDER BY trace_ts;
"""

from dotenv import load_dotenv
from langfuse import get_client
from psycopg.types.json import Jsonb

from rag.db import connect

load_dotenv()

# Plain Q&A spans only; agent traces have a different shape (multi-step, tool calls).
SPAN_NAMES = ["rag-ask", "rag-ask-stream"]


def spans(lf, name: str):
    """All observations with this name, paging the cursor-based v2 API."""
    cursor = None
    while True:
        res = lf.api.observations.get_many(
            name=name,
            fields="core,io,metadata",  # io/metadata are omitted unless asked for
            limit=50,
            cursor=cursor,
            request_options={"timeout_in_seconds": 60},
        )
        yield from res.data
        cursor = res.meta.cursor
        if not cursor:
            return


def sample() -> None:
    lf = get_client()
    with connect() as conn:
        added = 0
        for name in SPAN_NAMES:
            for o in spans(lf, name):
                if not o.input:
                    continue
                cur = conn.execute(
                    """INSERT INTO candidate_eval_questions
                           (trace_id, question, answer, retrieved, trace_ts)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (trace_id) DO NOTHING""",
                    (
                        o.trace_id,
                        str(o.input),
                        str(o.output) if o.output else None,
                        Jsonb((o.metadata or {}).get("retrieved", [])),
                        o.start_time,
                    ),
                )
                added += cur.rowcount
        pending = conn.execute(
            "SELECT count(*) AS n FROM candidate_eval_questions WHERE status = 'pending'"
        ).fetchone()["n"]
    print(f"added {added}, queue now {pending} pending")


if __name__ == "__main__":
    sample()
