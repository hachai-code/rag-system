"""Run the judge and persist it to Postgres (eval_runs + eval_results).

Companion to judge.py, which writes judgments.jsonl. Same narrow-per-dimension
judging and rubrics (imported from judge.py, not duplicated), but this records a run
(git SHA + config) and one row per question with its per-dimension scores/rationales
plus the run's real cost and latency.

Cost comes from token usage, which instructor only exposes via create_with_completion
(plain create() drops the raw response); latency is wall time over a question's judge
calls.

Schema lives in db/migrations/ (eval_runs, eval_results).

Run: uv run python -m evals.answer.judge_db [n]
"""

import subprocess
import sys
import time

from psycopg.types.json import Jsonb

from evals.answer.judge import (
    EVAL_FILE,
    IN_PRICE,
    JUDGE_MODEL,
    OUT_PRICE,
    eval_items,
    judge_client,
    judge_with_usage,
    rag_answer,
)
from rag.db import connect
from rag.query.answer import GEN_MODEL
from rag.query.retrieve import RELEVANCE_THRESHOLD, TOP_K


def git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return sha + ("-dirty" if dirty else "")


def main() -> None:
    items = eval_items()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(items)
    client = judge_client("openai-compat", JUDGE_MODEL)
    config = {
        "judge_model": JUDGE_MODEL,
        "mode": "tools",
        "eval_file": EVAL_FILE.name,
        "rag_model": GEN_MODEL,
        "top_k": TOP_K,
        "relevance_threshold": RELEVANCE_THRESHOLD,
    }

    with connect() as conn:
        run_id = conn.execute(
            "INSERT INTO eval_runs (git_sha, config) VALUES (%s, %s) RETURNING id",
            (git_sha(), Jsonb(config)),
        ).fetchone()["id"]
        conn.commit()

        judged = 0
        for item in items[:n]:
            try:
                ans = rag_answer(conn, item["question"])
                t0 = time.perf_counter()
                scores, rationales, in_tok, out_tok = {}, {}, 0, 0
                for code in item["axial_codes"]:
                    verdict, it, ot = judge_with_usage(
                        client, code, item["question"], ans, item["ideal_answer"]
                    )
                    scores[code], rationales[code] = verdict.passed, verdict.rationale
                    in_tok, out_tok = in_tok + it, out_tok + ot
                latency_ms = int((time.perf_counter() - t0) * 1000)
                cost = round(in_tok * IN_PRICE + out_tok * OUT_PRICE, 6)
            except Exception as e:
                # One bad item (network, parse failure past retries) shouldn't abort the run.
                print(f"  [skip] item {item['id']}: {type(e).__name__}: {e}")
                continue

            conn.execute(
                """INSERT INTO eval_results
                       (run_id, question_id, question, answer, scores, rationales, cost, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    run_id,
                    item["id"],
                    item["question"],
                    ans,
                    Jsonb(scores),
                    Jsonb(rationales),
                    cost,
                    latency_ms,
                ),
            )
            conn.commit()
            judged += 1
            marks = " ".join(f"{c}:{'P' if p else 'F'}" for c, p in scores.items())
            print(
                f"  [{judged:>2}/{n}] item {item['id']:>2}  {marks}  ${cost:.4f}  {latency_ms:>5}ms  {item['question'][:36]}"
            )

    print(f"\nrun {run_id}: judged {judged} item(s)")


if __name__ == "__main__":
    main()
