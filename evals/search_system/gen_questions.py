"""Generate synthetic eval questions from random corpus chunks.

Pulls N random chunks from the DB and asks Claude to write one question each
chunk answers. Because every question is generated *from* one chunk, that chunk is
a tight gold label (expected_chunk_ids = [chunk_id]) — the opposite of the broad
keyword-matched gold that inflates metrics.py today. The chunk's own text is kept
as the reference ("ideal") answer.

Output goes to evals/synthetic_questions.jsonl for review before merging the good
ones into eval_set.jsonl — these are drafts, not vetted ground truth.

Run: uv run python -m evals.gen_questions [n]
"""

import json
import sys
from pathlib import Path

import anthropic
import psycopg
from psycopg.rows import dict_row

from rag import DB_URL

OUT_FILE = Path(__file__).parent / "synthetic_questions.jsonl"
N_QUESTIONS = 50
MIN_CHARS = 300  # skip thin chunks (headings, stray lines) that can't anchor a question
# Haiku is cheap and fast, and writing one question from a passage is an easy task —
# the product answers use Sonnet, but generating eval inputs doesn't need it.
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You write evaluation questions for a retrieval system over the 'innerdance' "
    "corpus — transcripts of a body-based consciousness and meditation practice. "
    "Given one passage, write exactly ONE question that the passage directly "
    "answers. Rules: it must be answerable from this passage alone; specific enough "
    "that this passage is clearly the right source; phrased the way a student of "
    "innerdance would naturally ask it; never refer to 'the passage', 'the text', "
    "or 'the author'; not a yes/no question. Output only the question."
)


def random_chunks(conn: psycopg.Connection, n: int) -> list[dict]:
    return conn.execute(
        """SELECT c.id, d.title, c.content
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE length(c.content) > %s
           ORDER BY random() LIMIT %s""",
        (MIN_CHARS, n),
    ).fetchall()


def make_question(client: anthropic.Anthropic, content: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in response.content).strip()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_QUESTIONS
    client = anthropic.Anthropic()
    rows: list[dict] = []
    seen: set[int] = set()

    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        # Draw extra to cover chunks the model balks at — a bare references list
        # yields a refusal, not a question, which we skip. random() reshuffles each
        # call; `seen` avoids repeats and caps the loop so it can't run away.
        while len(rows) < n and len(seen) < 4 * n:
            for chunk in random_chunks(conn, n):
                if chunk["id"] in seen:
                    continue
                seen.add(chunk["id"])
                question = make_question(client, chunk["content"])
                # A real question ends with "?" and asks about the corpus; when the
                # model balks (e.g. a references chunk) it answers in the first
                # person ("I can't…", "I appreciate…"), so drop those too.
                if not question.endswith("?") or question.startswith(("I ", "I'")):
                    continue
                rows.append({
                    "id": len(rows) + 1,
                    "category": "synthetic",
                    "question": question,
                    "expected_chunk_ids": [chunk["id"]],
                    "ideal_answer": chunk["content"],
                    "source": {"chunk_id": chunk["id"], "title": chunk["title"]},
                })
                print(f"  [{len(rows):>2}/{n}] {chunk['title'][:28]:<28} {question[:58]}")
                if len(rows) == n:
                    break

    with OUT_FILE.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
