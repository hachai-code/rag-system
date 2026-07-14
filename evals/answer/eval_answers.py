"""End-to-end answer dataset for the innerdance RAG, for hand-grading.

For each random chunk: Sonnet writes a question the chunk answers, Opus writes the
reference ("correct") answer from that chunk, and the actual RAG app answers the
same question over the full corpus. Each row is written with an empty `grade` for
you to fill in by hand — compare `rag_answer` to `reference_answer` and mark it.

This covers the *answering* path end to end (retrieval + generation), where
search/ covers retrieval alone. The reference is grounded in the gold chunk,
and `retrieved_gold` records whether the app actually retrieved that chunk — so when
an answer is bad you can tell a retrieval miss from a generation failure.

Models: Sonnet writes questions; Opus writes the reference (strongest model = most
trustworthy gold); the app answers with its own model. No judge model — grading is
yours.

Runs are resumable: each row is appended and flushed as it completes, so a crash
loses at most the in-flight item. Re-running tops the file up to N, skipping chunks
already present — delete answer_feedback.jsonl to start fresh.

Run: uv run python -m evals.answer.eval_answers [n]
"""

import json
import sys
from pathlib import Path

import anthropic
import psycopg

from rag import RELEVANCE_THRESHOLD, answer, search
from rag.db import connect

OUT_FILE = Path(__file__).parent / "data" / "answer_feedback.jsonl"
N_ITEMS = 25  # each item is a Sonnet + an Opus + the app's own call, so the default is modest
PER_DOC = 10
MIN_CHARS = 300
GEN_MODEL = "claude-sonnet-4-6"
REF_MODEL = "claude-opus-4-8"
# Mirrors app.py's no-answer gate so the eval exercises the same refusal behaviour
# the API would. Replicated rather than imported from app.py to keep the FastAPI app
# and its Langfuse instrumentation out of an offline eval.
NO_ANSWER = "I don't have information on that in the innerdance corpus."

GEN_PROMPT = (
    "You write evaluation questions for a retrieval system over the 'innerdance' "
    "corpus — transcripts of a body-based consciousness and meditation practice, plus "
    "a neuroscience book it draws on. Given one passage, write exactly ONE question "
    "that the passage directly and fully answers.\n"
    "Rules:\n"
    "- Answerable from this passage alone, and specific enough that this passage is "
    "clearly the right source.\n"
    "- Third person only: never use 'you' or 'your', and do not address the speaker.\n"
    "- Do not rely on a proper name a reader wouldn't know (a patient's or student's "
    "name); ask about the concept or event instead.\n"
    "- Phrase it the way someone querying the corpus would naturally ask.\n"
    "- Not a yes/no question.\n"
    "Output only the question."
)

REF_PROMPT = (
    "Answer the question using only the information in the passage. Be accurate, "
    "specific, and complete, but add nothing the passage does not support. This is a "
    "reference answer used to grade another system, so it must be correct."
)


def candidate_chunks(conn: psycopg.Connection, per_doc: int) -> list[dict]:
    """Random chunks balanced across documents: at most `per_doc` from each, then
    interleaved so one large document can't dominate the sample."""
    return conn.execute(
        """SELECT id, title, content FROM (
               SELECT c.id, d.title, c.content,
                      row_number() OVER (PARTITION BY c.document_id ORDER BY random()) AS rn
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE length(c.content) > %s
           ) ranked
           WHERE rn <= %s
           ORDER BY rn, random()""",
        (MIN_CHARS, per_doc),
    ).fetchall()


def make_question(client: anthropic.Anthropic, content: str) -> str:
    response = client.messages.create(
        model=GEN_MODEL, max_tokens=100, system=GEN_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in response.content).strip()


def reference_answer(client: anthropic.Anthropic, question: str, content: str) -> str:
    response = client.messages.create(
        model=REF_MODEL, max_tokens=512, system=REF_PROMPT,
        messages=[{"role": "user", "content": f"Passage:\n{content}\n\nQuestion: {question}"}],
    )
    return "".join(block.text for block in response.content).strip()


def rag_answer(conn: psycopg.Connection, question: str) -> tuple[str, list[dict]]:
    """Answer exactly as app.py's /ask does: vector search, the relevance gate, then
    generate over the retrieved chunks."""
    hits = search(conn, question)
    if not hits or hits[0]["distance"] > RELEVANCE_THRESHOLD:
        return NO_ANSWER, hits
    text, _ = answer(question, hits)
    return text, hits


def existing_rows() -> list[dict]:
    """Rows already written, so a re-run can resume rather than start over."""
    if not OUT_FILE.exists():
        return []
    return [json.loads(line) for line in OUT_FILE.read_text().splitlines() if line.strip()]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_ITEMS
    # max_retries above the default 2 gives the question/reference calls extra headroom
    # on transient 429/5xx during a long run (the RAG answer call retries on its own).
    client = anthropic.Anthropic(max_retries=4)
    rows = existing_rows()
    done = {r["source"]["chunk_id"] for r in rows}

    with connect() as conn, OUT_FILE.open("a") as out:
        for chunk in candidate_chunks(conn, PER_DOC):
            if len(rows) >= n:
                break
            if chunk["id"] in done:
                continue
            try:
                question = make_question(client, chunk["content"])
                if not question.endswith("?") or question.startswith(("I ", "I'")):
                    continue  # model balked; not a real failure, just skip it
                reference = reference_answer(client, question, chunk["content"])
                rag_text, hits = rag_answer(conn, question)
            except Exception as e:
                # One bad item (network, rate limit past retries) shouldn't abort the run.
                print(f"  [skip] chunk {chunk['id']}: {type(e).__name__}: {e}")
                continue
            retrieved_ids = [h["id"] for h in hits]
            row = {
                "id": len(rows) + 1,
                "question": question,
                "source": {"chunk_id": chunk["id"], "title": chunk["title"]},
                "reference_answer": reference,
                "rag_answer": rag_text,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_gold": chunk["id"] in retrieved_ids,
                "grade": "",  # fill in by hand, e.g. up / down (or correct / partial / wrong)
            }
            rows.append(row)
            done.add(chunk["id"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()  # so a crash keeps every completed row
            gold = "gold" if chunk["id"] in retrieved_ids else "miss"
            print(f"  [{len(rows):>2}/{n}] {gold} {chunk['title'][:24]:<24} {question[:48]}")

    got = sum(r["retrieved_gold"] for r in rows)
    print(f"\n{OUT_FILE} has {len(rows)} rows  ·  gold chunk retrieved in {got}/{len(rows)}")


if __name__ == "__main__":
    main()
