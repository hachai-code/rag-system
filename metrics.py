"""Retrieval metrics for the RAG baseline: recall@5 and MRR.

recall@5 — fraction of graded questions whose relevant chunk appears in the top 5.
MRR      — mean of 1/rank of the first relevant chunk (0 if absent from the top 5).

Gold labels are eval_set.jsonl's `expected_chunk_ids`; questions without any are
skipped (no-answer / unlabeled). Each run is appended to metrics_log.jsonl so you
can watch the number move as you improve retrieval.

Run: uv run metrics.py [label]
"""

import datetime
import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from rag import DB_URL, search

K = 5
EVAL_FILE = Path("eval_set.jsonl")
LOG_FILE = Path("metrics_log.jsonl")


def recall_at_k(retrieved: list[int], expected: set[int], k: int = K) -> float:
    """Did any relevant chunk land in the top k?"""
    return 1.0 if set(retrieved[:k]) & expected else 0.0


def reciprocal_rank(retrieved: list[int], expected: set[int], k: int = K) -> float:
    """1 / rank of the first relevant chunk; 0 if none in the top k."""
    for rank, chunk_id in enumerate(retrieved[:k], 1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "naive"
    rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    graded = [r for r in rows if r.get("expected_chunk_ids")]

    recalls, rrs = [], []
    print(f"{'id':>3}  {'category':<11} {'gold':>4} {'hit':>3} {'rank':>4}")
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        for r in graded:
            retrieved = [h["id"] for h in search(conn, r["question"])]
            expected = set(r["expected_chunk_ids"])
            recalls.append(recall_at_k(retrieved, expected))
            rrs.append(reciprocal_rank(retrieved, expected))
            rank = next((i for i, cid in enumerate(retrieved[:K], 1) if cid in expected), None)
            print(f"{r['id']:>3}  {r['category']:<11} {len(expected):>4} {'Y' if recalls[-1] else '.':>3} {rank or '-':>4}")

    recall = sum(recalls) / len(recalls)
    mrr = sum(rrs) / len(rrs)
    print(f"\n{label} RAG (n={len(graded)}):  recall@{K} = {recall:.2f}   MRR = {mrr:.2f}")

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "n": len(graded),
        "recall_at_5": round(recall, 4),
        "mrr": round(mrr, 4),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Recorded to {LOG_FILE}")


if __name__ == "__main__":
    main()
