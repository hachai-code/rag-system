"""Populate eval_set.jsonl with draft RAG answers + retrieved sources for review.

Runs every question through the RAG (retrieve + Claude answer) and writes the
results back as `draft_answer` and `top_sources`. Re-runnable; overwrites the
draft fields. These drafts are a starting point for a human to correct — they
come from the system under test, so they are NOT ground truth.

Run: uv run gen_eval.py
"""

import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from rag import DB_URL, answer, search

EVAL_FILE = Path("eval_set.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]

    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        for row in rows:
            hits = search(conn, row["question"])
            row["top_sources"] = [
                {"title": h["title"][:45], "distance": round(h["distance"], 4)} for h in hits
            ]
            row["draft_answer"] = answer(row["question"], hits)
            print(f"  [{row['id']:>2}/{len(rows)}] {row['category']:<11} {row['question'][:45]}")

    with EVAL_FILE.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} draft answers to {EVAL_FILE}")


if __name__ == "__main__":
    main()
