"""Retrieval metrics for the RAG baseline: recall@5 and MRR.

recall@5 — fraction of graded questions whose relevant chunk appears in the top 5.
MRR      — mean of 1/rank of the first relevant chunk (0 if absent from the top 5).

Gold labels are eval_set.jsonl's `expected_chunk_ids`; questions without any are
skipped (no-answer / unlabeled). Each run is appended to metrics_log.jsonl so you
can watch the number move as you improve retrieval.

Run: uv run python -m evals.search.metrics [label]
"""

import datetime
import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from rag import DB_URL, hybrid_search, rerank_search, search

K = 5
EVAL_FILE = Path(__file__).parent.parent / "eval_set.jsonl"
LOG_FILE = Path(__file__).parent / "data" / "metrics_log.jsonl"


def recall_at_k(retrieved: list[int], expected: set[int], k: int = K) -> float:
    """Did any relevant chunk land in the top k?"""
    return 1.0 if set(retrieved[:k]) & expected else 0.0


def reciprocal_rank(retrieved: list[int], expected: set[int], k: int = K) -> float:
    """1 / rank of the first relevant chunk; 0 if none in the top k."""
    for rank, chunk_id in enumerate(retrieved[:k], 1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def gold_ids(conn: psycopg.Connection, keywords: list[str]) -> set[int]:
    """Chunks (in the CURRENT db) whose content contains all keywords. Recomputed
    each run so the gold set tracks re-embeds/re-chunks instead of frozen IDs."""
    where = " AND ".join(["content ILIKE %s"] * len(keywords))
    sql = f"SELECT id FROM chunks WHERE {where}"
    return {row["id"] for row in conn.execute(sql, [f"%{k}%" for k in keywords]).fetchall()}


def aggregate(results: list[tuple[float, float]]) -> tuple[float, float]:
    recall = sum(rec for rec, _ in results) / len(results)
    mrr = sum(rr for _, rr in results) / len(results)
    return recall, mrr


def main() -> None:
    flags = sys.argv[1:]
    if "--rerank" in flags:
        retriever, default_label = rerank_search, "rerank"
    elif "--hybrid" in flags:
        retriever, default_label = hybrid_search, "hybrid"
    else:
        retriever, default_label = search, "naive"
    positional = [a for a in flags if not a.startswith("-")]
    label = positional[0] if positional else default_label
    rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    graded = [r for r in rows if r.get("relevance_keywords")]

    print(f"{'id':>3}  {'category':<11} {'gold':>4} {'hit':>3} {'rank':>4}")
    scored = []  # (gold_count, recall, rr)
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        for r in graded:
            expected = gold_ids(conn, r["relevance_keywords"])
            retrieved = [h["id"] for h in retriever(conn, r["question"])]
            rec, rr = recall_at_k(retrieved, expected), reciprocal_rank(retrieved, expected)
            scored.append((len(expected), rec, rr))
            rank = next((i for i, cid in enumerate(retrieved[:K], 1) if cid in expected), None)
            print(f"{r['id']:>3}  {r['category']:<11} {len(expected):>4} {'Y' if rec else '.':>3} {rank or '-':>4}")

    # Tight gold (<=6 specific chunks) is the discriminating subset; the full set
    # is inflated by broad keyword-labeled gold.
    full = [(rec, rr) for _, rec, rr in scored]
    tight = [(rec, rr) for gold_count, rec, rr in scored if gold_count <= 6]
    recall, mrr = aggregate(full)
    t_recall, t_mrr = aggregate(tight)

    print(f"\n{label}:")
    print(f"  all graded (n={len(full)}):   recall@{K} = {recall:.2f}   MRR = {mrr:.2f}")
    print(f"  tight gold (n={len(tight)}):   recall@{K} = {t_recall:.2f}   MRR = {t_mrr:.2f}")

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": label, "n": len(full),
        "recall_at_5": round(recall, 4), "mrr": round(mrr, 4),
        "tight_n": len(tight),
        "tight_recall_at_5": round(t_recall, 4), "tight_mrr": round(t_mrr, 4),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Recorded to {LOG_FILE}")


if __name__ == "__main__":
    main()
