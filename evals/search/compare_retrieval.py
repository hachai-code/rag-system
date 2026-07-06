"""Compare keyword (full-text) vs vector retrieval on the queries vector handles
worst.

"Worst for vector" = highest top-1 cosine distance, i.e. the queries where the
nearest chunk is still far away and vector search is least confident. (Gold rank
can't rank the queries here: the eval's gold sets are broad, so vector hits rank
1 almost everywhere — see failure-analysis.md, which used distance for the same
reason.) For the N least-confident graded questions we show where vector and
keyword each rank the gold chunk, to see which queries lexical search rescues.

Run: uv run python -m evals.search.compare_retrieval
"""

import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from rag import DB_URL, keyword_search, search

from evals.search.metrics import gold_ids

K = 10        # retrieve this deep so a poor rank is still visible (eval scores @5)
N_HARD = 10   # how many least-confident-for-vector queries to inspect
EVAL_FILE = Path(__file__).parent.parent / "eval_set.jsonl"


def gold_rank(hits: list[dict], gold: set[int]) -> int | None:
    """1-based rank of the first gold chunk in hits, or None if absent."""
    for rank, hit in enumerate(hits, 1):
        if hit["id"] in gold:
            return rank
    return None


def fmt(rank: int | None) -> str:
    return str(rank) if rank else "miss"


def main() -> None:
    rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    graded = [r for r in rows if r.get("relevance_keywords")]

    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        scored = []  # (question_row, gold_ids, vector_hits)
        for r in graded:
            gold = gold_ids(conn, r["relevance_keywords"])
            scored.append((r, gold, search(conn, r["question"], K)))

        # Least confident first: largest top-1 distance.
        scored.sort(key=lambda s: s[2][0]["distance"], reverse=True)
        hard = scored[:N_HARD]

        print(f"{'id':>3}  {'category':<11} {'dist':>5} {'vec':>4} {'kw':>4}  question")
        kw_only = 0  # gold keyword found but vector missed
        for r, gold, v_hits in hard:
            v_rank = gold_rank(v_hits, gold)
            k_rank = gold_rank(keyword_search(conn, r["question"], K), gold)
            if k_rank and not v_rank:
                kw_only += 1
            print(
                f"{r['id']:>3}  {r['category']:<11} {v_hits[0]['distance']:>5.2f} "
                f"{fmt(v_rank):>4} {fmt(k_rank):>4}  {r['question'][:60]}"
            )

    print(f"\nOn these {len(hard)} hardest-for-vector queries, keyword found a gold "
          f"chunk that vector missed on {kw_only}.")
    print("vec/kw = rank of first gold chunk in each method's top 10 ('miss' = not in top 10).")


if __name__ == "__main__":
    main()
