"""Generate synthetic eval questions from random corpus chunks.

For each chunk, Sonnet writes one question the chunk answers. Because every question
is generated *from* one chunk, that chunk is a tight gold label (expected_chunk_ids
= [chunk_id]) — the opposite of the broad keyword-matched gold that inflates
metrics.py today. The chunk's own text is kept as the reference ("ideal") answer.

Two guards keep the candidates clean:
- Chunks are sampled evenly across documents, so the large neuroscience book can't
  drown out the innerdance transcripts.
- The prompt forbids second-person phrasing and obscure proper names, so questions
  read like real corpus queries.

Whether a chunk actually answers its question (grounding) is left to human review —
these are drafts to grade, not vetted ground truth. Output goes to
synthetic_questions.jsonl; merge the good ones into eval_set.jsonl after grading.

Run: uv run python -m evals.search.gen_questions [n]
"""

import json
import sys
from pathlib import Path

import anthropic
import psycopg
from psycopg.rows import dict_row

from rag import DB_URL

OUT_FILE = Path(__file__).parent / "data" / "synthetic_questions.jsonl"
N_QUESTIONS = 50
PER_DOC = 10  # candidate cap per document; the pool is drawn larger than N to absorb skips
MIN_CHARS = 300  # skip thin chunks (headings, stray lines) that can't anchor a question
# Sonnet, a step up from Haiku, for higher-quality questions. Generation is the only
# model step — grounding is verified by hand, so there's no judge model here.
GEN_MODEL = "claude-sonnet-4-6"

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


def candidate_chunks(conn: psycopg.Connection, per_doc: int) -> list[dict]:
    """Random chunks balanced across documents: at most `per_doc` from each, then
    interleaved (one per document before a second from any) so one large document
    can't dominate the sample."""
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
        model=GEN_MODEL,
        max_tokens=100,
        system=GEN_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in response.content).strip()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_QUESTIONS
    client = anthropic.Anthropic()
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        candidates = candidate_chunks(conn, PER_DOC)

    rows: list[dict] = []
    skipped = 0
    for chunk in candidates:
        if len(rows) == n:
            break
        question = make_question(client, chunk["content"])
        # A real question ends with "?" and asks about the corpus; when the model
        # balks (e.g. a references chunk) it answers in the first person, so skip those.
        if not question.endswith("?") or question.startswith(("I ", "I'")):
            skipped += 1
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

    with OUT_FILE.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} questions to {OUT_FILE}  (skipped {skipped} malformed)")


if __name__ == "__main__":
    main()
