"""One-off backfill: copy plain /ask/stream answers from Langfuse into Q&A memory.

Until /ask/stream started writing to qa_memory, its answers never reached memory, but
each one still landed in Langfuse as a `rag-ask-stream` span (question as input, full
answer as output; see rag/app.py). This replays those spans through save_qa_record.

The spans carry no user id, so every backfilled memory goes to the one user named by
--email. Only run it as-is if that user asked all those questions.

Safe to re-run: each question goes through the same near-duplicate check as a live run,
so ones already in that user's memory (including ones inserted earlier in this run) are
skipped. --apply costs one Voyage embedding per distinct question; the dry run makes no
Voyage calls, so it lists every distinct answered question without the memory check.

    uv run python -m db.backfill_qa_memory --email you@example.com          # dry run
    uv run python -m db.backfill_qa_memory --email you@example.com --apply
"""

import argparse
import json

from dotenv import load_dotenv
from langfuse import get_client

from rag.db import connect
from rag.guardrails import BLOCKED
from rag.query.gate import NO_ANSWER
from rag.query.memory import QA_DEDUP_SCORE, lookup_similar_qa, save_qa_record

load_dotenv()

SPAN_NAME = "rag-ask-stream"


def spans(lf):
    """Every rag-ask-stream observation, paging the cursor-based v2 API (order unspecified)."""
    cursor = None
    while True:
        res = lf.api.observations.get_many(
            name=SPAN_NAME,
            fields="core,io",
            limit=100,
            cursor=cursor,
            request_options={"timeout_in_seconds": 60},
        )
        yield from res.data
        cursor = res.meta.cursor
        if not cursor:
            return


def _text(value) -> str:
    """Span IO as plain text. The API returns IO as raw strings; a str the SDK
    JSON-encoded comes back quoted, so unwrap that."""
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return str(value).strip()


def backfill(email: str, apply: bool) -> None:
    lf = get_client()
    with connect() as conn:
        row = conn.execute('SELECT id FROM "user" WHERE email = %s', (email,)).fetchone()
        if row is None:
            raise SystemExit(f"no user with email {email!r}")
        user_id = row["id"]

        # Oldest first, so when a question was asked twice the earlier answer is kept.
        candidates = sorted(
            ((o.start_time, _text(o.input), _text(o.output)) for o in spans(lf)),
            key=lambda c: c[0],
        )
        saved = skipped_dup = skipped_empty = 0
        seen: set[str] = set()  # exact repeats skip the embedding call entirely
        for start_time, question, answer in candidates:
            # No output = the stream died before finishing; NO_ANSWER / BLOCKED are the
            # gate and input rail, which a live run never saves either.
            if not question or not answer or answer in (NO_ANSWER, BLOCKED):
                skipped_empty += 1
                continue
            if question in seen:
                skipped_dup += 1
                continue
            seen.add(question)
            if apply:
                match = lookup_similar_qa(conn, question, user_id, limit=1)
                if match.top_score >= QA_DEDUP_SCORE:
                    skipped_dup += 1
                    continue
                save_qa_record(conn, question, answer, [], [], match, user_id, start_time)
            saved += 1
            print(f"{'saved' if apply else 'candidate'}: {start_time:%Y-%m-%d} {question[:80]}")

    print(
        f"\n{len(candidates)} spans: {saved} {'saved' if apply else 'candidates'},"
        f" {skipped_dup} duplicates, {skipped_empty} unanswered/blocked/incomplete"
    )
    if not apply:
        print("dry run (no memory check, no writes) — re-run with --apply to write")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True, help="user who owns the backfilled memories")
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args()
    backfill(args.email, args.apply)
