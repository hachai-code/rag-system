"""One-command, reproducible eval runs.

    uv run python -m evals.run --config evals/configs/baseline.json

Runs the full suite (retrieve -> answer -> judge) against a versioned config, records
the run in Postgres (eval_runs + eval_results), and prints a summary table with the
per-dimension / cost / latency delta against the previous run.

The config is a JSON file holding the tunable knobs (retrieval params, generation
model + prompt, judge model). Everything that shapes a run is fingerprinted into a
content hash stored on the run, so a run is reproducible from its config + git SHA:
change a knob or the prompt and the hash changes; leave them and it doesn't.

Reuses the judge from answer_system/ (rubrics, per-dimension Opus calls, token-priced
cost). A new config is a new file — no code edits — which is the point.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from evals.answer_system.judge import NO_ANSWER, RUBRICS, SYSTEM, eval_items
from evals.answer_system.judge_db import IN_PRICE, OUT_PRICE, git_sha, judge_client, judge_with_usage
from rag import DB_URL, SYSTEM_PROMPT, answer, rerank_search, search


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_config(path: str) -> tuple[dict, str]:
    """Returns (stored_config, generation_prompt). stored_config is what goes in the run
    record: the knobs plus a content hash of everything that affects the run."""
    cfg = json.loads(Path(path).read_text())
    ret, gen, jud = cfg["retrieval"], cfg["generation"], cfg["judge"]

    # Generation prompt: a file path (a versioned variant) or null = the production prompt.
    prompt_file = gen.get("prompt_file")
    gen_prompt = (Path(path).parent / prompt_file).read_text() if prompt_file else SYSTEM_PROMPT
    prompt_sha = sha(gen_prompt)
    rubric_sha = sha(SYSTEM + json.dumps(RUBRICS, sort_keys=True))

    # The fingerprint: hash every input that changes what a run produces.
    fingerprint = json.dumps({
        "top_k": ret["top_k"],
        "relevance_threshold": ret["relevance_threshold"],
        "gen_provider": gen["provider"],
        "gen_model": gen["model"],
        "prompt_sha": prompt_sha,
        "judge_provider": jud["provider"],
        "judge_model": jud["model"],
        "rubric_sha": rubric_sha,
    }, sort_keys=True)

    stored = {
        "name": cfg.get("name", Path(path).stem),
        "hash": sha(fingerprint)[:12],
        "retrieval": ret,
        "generation": {"provider": gen["provider"], "model": gen["model"],
                       "prompt_file": prompt_file, "prompt_sha": prompt_sha[:12]},
        "judge": {"provider": jud["provider"], "model": jud["model"], "rubric_sha": rubric_sha[:12]},
    }
    return stored, gen_prompt


def evaluate(conn, client, items, top_k, threshold, gen_provider, gen_model, gen_prompt):
    """Run retrieve -> answer -> judge for each item; yield one result dict per item.
    Skips (and logs) an item whose calls fail so one bad item can't abort the run.
    Shared by run.py (persist + delta) and check_regression.py (the CI gate)."""
    for item in items:
        try:
            gate = search(conn, item["question"], k=1)  # cheap coverage check only
            if not gate or gate[0]["distance"] > threshold:
                ans = NO_ANSWER
            else:
                hits = rerank_search(conn, item["question"], k=top_k)  # hybrid + RRF + rerank
                ans, _ = answer(item["question"], hits, model=gen_model, system=gen_prompt, provider=gen_provider)
            t0 = time.perf_counter()
            scores, rationales, in_tok, out_tok = {}, {}, 0, 0
            for code in item["axial_codes"]:
                verdict, it, ot = judge_with_usage(client, code, item["question"], ans, item["ideal_answer"])
                scores[code], rationales[code] = verdict.passed, verdict.rationale
                in_tok, out_tok = in_tok + it, out_tok + ot
            latency_ms = int((time.perf_counter() - t0) * 1000)
            cost = round(in_tok * IN_PRICE + out_tok * OUT_PRICE, 6)
        except Exception as e:
            print(f"  [skip] item {item['id']}: {type(e).__name__}: {e}")
            continue
        yield {"id": item["id"], "question": item["question"], "answer": ans,
               "scores": scores, "rationales": rationales, "cost": cost, "latency_ms": latency_ms}


def run_summary(conn, run_id) -> dict:
    """Per-dimension pass rate, total cost, and mean latency for one run."""
    rows = conn.execute(
        "SELECT scores, cost, latency_ms FROM eval_results WHERE run_id = %s", (run_id,)
    ).fetchall()
    per_dim = defaultdict(list)
    for r in rows:
        for dim, passed in r["scores"].items():
            per_dim[dim].append(passed)
    return {
        "n": len(rows),
        "pass_rate": {dim: sum(v) / len(v) for dim, v in per_dim.items()},
        "cost": float(sum(r["cost"] for r in rows)),
        "latency": sum(r["latency_ms"] for r in rows) / len(rows) if rows else 0.0,
    }


def _delta(cur: float, prev: float | None, fmt) -> str:
    if prev is None:
        return f"{fmt(cur):>9}{'—':>9}{'—':>9}"
    d = cur - prev
    return f"{fmt(cur):>9}{fmt(prev):>9}{('+' if d >= 0 else '') + fmt(d):>9}"


def print_summary(conn, run_id, prev_id, cfg) -> None:
    cur = run_summary(conn, run_id)
    prev = run_summary(conn, prev_id) if prev_id else None
    pct = lambda x: f"{x:.2f}"

    print(f"\nconfig: {cfg['name']}  (hash {cfg['hash']})")
    print(f"run {run_id}" + (f" vs run {prev_id}\n" if prev_id else "  (no previous run)\n"))
    print(f"{'metric':<14}{'this':>9}{'prev':>9}{'Δ':>9}")

    dims = sorted(set(cur["pass_rate"]) | (set(prev["pass_rate"]) if prev else set()))
    for dim in dims:
        cur_rate = cur["pass_rate"].get(dim)
        if cur_rate is None:  # dimension in a previous run but not exercised in this one
            continue
        prev_rate = prev["pass_rate"].get(dim) if prev else None
        print(f"{dim:<14}" + _delta(cur_rate, prev_rate, pct))

    print(f"{'questions':<14}{cur['n']:>9}{(prev['n'] if prev else '—'):>9}")
    print(f"{'cost':<14}" + _delta(cur["cost"], prev["cost"] if prev else None, lambda x: f"${x:.4f}"))
    print(f"{'latency':<14}" + _delta(cur["latency"], prev["latency"] if prev else None, lambda x: f"{x:.0f}ms"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a config JSON (see evals/configs/)")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, help="judge only the first N items (quick check)")
    args = ap.parse_args()

    cfg, gen_prompt = load_config(args.config)
    items = eval_items()
    if args.split != "all":
        items = [it for it in items if it["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    top_k = cfg["retrieval"]["top_k"]
    threshold = cfg["retrieval"]["relevance_threshold"]
    gen_provider = cfg["generation"]["provider"]
    gen_model = cfg["generation"]["model"]
    client = judge_client(cfg["judge"]["provider"], cfg["judge"]["model"])

    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        prev_id = conn.execute("SELECT max(id) AS id FROM eval_runs").fetchone()["id"]
        run_id = conn.execute(
            "INSERT INTO eval_runs (git_sha, config) VALUES (%s, %s) RETURNING id",
            (git_sha(), Jsonb(cfg)),
        ).fetchone()["id"]
        conn.commit()

        judged = 0
        for r in evaluate(conn, client, items, top_k, threshold, gen_provider, gen_model, gen_prompt):
            conn.execute(
                """INSERT INTO eval_results
                       (run_id, question_id, question, answer, scores, rationales, cost, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (run_id, r["id"], r["question"], r["answer"],
                 Jsonb(r["scores"]), Jsonb(r["rationales"]), r["cost"], r["latency_ms"]),
            )
            conn.commit()
            judged += 1
            marks = " ".join(f"{c}:{'P' if p else 'F'}" for c, p in r["scores"].items())
            print(f"  [{judged:>2}/{len(items)}] item {r['id']:>2}  {marks}  ${r['cost']:.4f}  {r['question'][:36]}")

        print_summary(conn, run_id, prev_id, cfg)


if __name__ == "__main__":
    main()
