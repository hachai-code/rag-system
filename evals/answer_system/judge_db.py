"""Run the judge and persist it to Postgres (eval_runs + eval_results).

Companion to judge.py, which writes judgments.jsonl. Same narrow-per-dimension Opus
judging and rubrics (imported from judge.py, not duplicated), but this records a run
(git SHA + config) and one row per question with its per-dimension scores/rationales
plus the run's real cost and latency.

Cost comes from token usage, which instructor only exposes via create_with_completion
(plain create() drops the raw response); latency is wall time over a question's judge
calls. Opus 4.8 list price is $5 / $25 per 1M input/output tokens (platform.claude.com).

Schema lives in db/migrations/ (eval_runs, eval_results).

Run: uv run python -m evals.answer_system.judge_db [n]
"""

import os
import subprocess
import sys
import time

import instructor
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from evals.answer_system.judge import (
    EVAL_FILE, JUDGE_MODEL, RUBRICS, SYSTEM, Verdict, eval_items, rag_answer,
)
from rag import DB_URL, GEN_MODEL, OPENROUTER_BASE_URL, RELEVANCE_THRESHOLD, TOP_K

# Judge token price (DeepSeek V4 Pro on OpenRouter): $0.435 / $0.87 per 1M in/out.
IN_PRICE, OUT_PRICE = 0.435 / 1_000_000, 0.87 / 1_000_000


def judge_client(provider: str, model: str):
    """Build the instructor judge client for the configured provider: Anthropic native,
    or an OpenAI-compatible model (e.g. DeepSeek V4 Pro via OpenRouter). Both fill the
    Verdict via structured output, so judge_with_usage doesn't care which it got."""
    if provider == "openai-compat":
        # Mode.JSON, not TOOLS: DeepSeek on OpenRouter is unreliable at tool-calling, so
        # ask for the Verdict as JSON in the content (matches the generation path in rag.py).
        return instructor.from_provider(
            f"openai/{model}",
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            mode=instructor.Mode.JSON,
        )
    return instructor.from_provider(f"anthropic/{model}", mode=instructor.Mode.TOOLS)


def judge_with_usage(client, code: str, question: str, answer_text: str, ideal: str = ""):
    """judge.judge_dimension, but via create_with_completion so we also get token usage.
    Returns (Verdict, input_tokens, output_tokens)."""
    name, criterion, pass_def, fail_def = RUBRICS[code]
    user = f"Question:\n{question}\n\nAnswer to judge:\n{answer_text}"
    if ideal:
        user += f"\n\nReference answer (ground truth):\n{ideal}"
    verdict, completion = client.create_with_completion(
        max_tokens=300,
        max_retries=2,
        response_model=Verdict,
        messages=[
            {"role": "system", "content": SYSTEM.format(
                name=name, criterion=criterion, pass_def=pass_def, fail_def=fail_def)},
            {"role": "user", "content": user},
        ],
    )
    # Anthropic usage exposes input_tokens/output_tokens; OpenAI-compatible (OpenRouter)
    # exposes prompt_tokens/completion_tokens — accept either so one helper serves both judges.
    usage = completion.usage
    in_tok = getattr(usage, "input_tokens", None) or usage.prompt_tokens
    out_tok = getattr(usage, "output_tokens", None) or usage.completion_tokens
    return verdict, in_tok, out_tok


def git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return sha + ("-dirty" if dirty else "")


def main() -> None:
    items = eval_items()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(items)
    client = instructor.from_provider(f"anthropic/{JUDGE_MODEL}", mode=instructor.Mode.TOOLS)
    config = {
        "judge_model": JUDGE_MODEL,
        "mode": "tools",
        "eval_file": EVAL_FILE.name,
        "rag_model": GEN_MODEL,
        "top_k": TOP_K,
        "relevance_threshold": RELEVANCE_THRESHOLD,
    }

    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
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
                    verdict, it, ot = judge_with_usage(client, code, item["question"], ans, item["ideal_answer"])
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
                (run_id, item["id"], item["question"], ans,
                 Jsonb(scores), Jsonb(rationales), cost, latency_ms),
            )
            conn.commit()
            judged += 1
            marks = " ".join(f"{c}:{'P' if p else 'F'}" for c, p in scores.items())
            print(f"  [{judged:>2}/{n}] item {item['id']:>2}  {marks}  ${cost:.4f}  {latency_ms:>5}ms  {item['question'][:36]}")

    print(f"\nrun {run_id}: judged {judged} item(s)")


if __name__ == "__main__":
    main()
