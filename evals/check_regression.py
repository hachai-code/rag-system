"""CI regression gate: fail the build if a category's pass rate drops too far.

    uv run python -m evals.check_regression --config evals/configs/baseline.json
    uv run python -m evals.check_regression --config ... --update-baseline   # bless

Runs the suite for a config, computes per-dimension pass rate, and compares it to the
committed baseline (evals/baseline_metrics.json). Exits non-zero if any well-sampled
category dropped more than the threshold — that exit code is what blocks a PR.

Two deliberate guards against false alarms, because judge pass rates are noisy:
  - THRESHOLD (0.15): a category must drop by more than 15 points to count. Normal
    run-to-run variance is smaller; a real prompt/retrieval regression is larger.
  - MIN_SAMPLES (5): categories with fewer than 5 judged items are reported but never
    fail the build — a 1-item category swings 0–100% on a single flip, so gating it
    would be noise, not signal. Today only C and D clear this bar (see failure-taxonomy.md);
    grow A/B/E before trusting them.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from evals.answer_system.judge import eval_items
from evals.answer_system.judge_db import judge_client
from evals.run import evaluate, load_config
from rag import ANSWER_FORMAT, DB_URL

BASELINE = Path(__file__).parent / "baseline_metrics.json"
THRESHOLD = 0.15
MIN_SAMPLES = 5
MIN_COMPLETE = 0.9  # if more than 10% of items fail to judge (API outage, credits), the
                    # run can't gate quality — error the build instead of guessing.


def measure(config_path: str, split: str, limit: int | None) -> tuple[dict, dict, int, int]:
    """Run the suite; return (pass_rate, counts, attempted, succeeded). attempted vs
    succeeded lets the caller refuse to gate on a half-completed run."""
    cfg, gen_prompt = load_config(config_path)
    items = eval_items()
    if split != "all":
        items = [it for it in items if it["split"] == split]
    if limit:
        items = items[:limit]

    client = judge_client(cfg["judge"]["provider"], cfg["judge"]["model"])
    per_dim = defaultdict(list)
    succeeded = 0
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        for r in evaluate(conn, client,
                          items, cfg["retrieval"]["top_k"], cfg["retrieval"]["relevance_threshold"],
                          cfg["retrieval"].get("method", "rerank"),
                          cfg["retrieval"].get("query_enhancement"),
                          cfg["retrieval"].get("parent_document", False),
                          cfg["generation"]["provider"], cfg["generation"]["model"], gen_prompt,
                          cfg["generation"].get("format", ANSWER_FORMAT)):
            succeeded += 1
            for dim, passed in r["scores"].items():
                per_dim[dim].append(passed)
    pass_rate = {dim: sum(v) / len(v) for dim, v in per_dim.items()}
    counts = {dim: len(v) for dim, v in per_dim.items()}
    return pass_rate, counts, len(items), succeeded


def find_regressions(baseline: dict, current: dict, threshold: float, min_samples: int) -> list[dict]:
    """Pure comparison: a dimension regresses if it had enough baseline samples and its
    pass rate dropped by more than `threshold`. Returns one record per regression."""
    out = []
    for dim, base_rate in baseline["pass_rate"].items():
        if baseline["counts"].get(dim, 0) < min_samples:
            continue
        cur_rate = current.get(dim, 0.0)
        if base_rate - cur_rate > threshold:
            out.append({"dim": dim, "baseline": base_rate, "current": cur_rate, "drop": base_rate - cur_rate})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--update-baseline", action="store_true", help="write baseline_metrics.json and exit")
    args = ap.parse_args()

    pass_rate, counts, attempted, succeeded = measure(args.config, args.split, args.limit)

    if attempted and succeeded / attempted < MIN_COMPLETE:
        print(f"ERROR: only {succeeded}/{attempted} items judged "
              f"(<{MIN_COMPLETE:.0%}) — eval incomplete, not gating. Check API/DB.")
        sys.exit(2)

    if args.update_baseline:
        cfg, _ = load_config(args.config)
        BASELINE.write_text(json.dumps(
            {"config_hash": cfg["hash"], "split": args.split, "pass_rate": pass_rate, "counts": counts},
            indent=2) + "\n")
        print(f"wrote baseline ({args.split} split, config {cfg['hash']}): "
              + ", ".join(f"{d}={pass_rate[d]:.2f}(n={counts[d]})" for d in sorted(pass_rate)))
        return

    baseline = json.loads(BASELINE.read_text())
    regressions = find_regressions(baseline, pass_rate, args.threshold, MIN_SAMPLES)

    print(f"{'dim':<5}{'baseline':>10}{'current':>9}{'n':>5}  status")
    for dim in sorted(set(baseline["pass_rate"]) | set(pass_rate)):
        base = baseline["pass_rate"].get(dim)
        cur = pass_rate.get(dim)
        n = baseline["counts"].get(dim, 0)
        status = "gated" if n >= MIN_SAMPLES else f"too small (n<{MIN_SAMPLES})"
        if any(r["dim"] == dim for r in regressions):
            status = "REGRESSION"
        bs = f"{base:.2f}" if base is not None else "—"
        cs = f"{cur:.2f}" if cur is not None else "—"
        print(f"{dim:<5}{bs:>10}{cs:>9}{n:>5}  {status}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} category(ies) dropped >{args.threshold:.2f}:")
        for r in regressions:
            print(f"  {r['dim']}: {r['baseline']:.2f} -> {r['current']:.2f}  (-{r['drop']:.2f})")
        sys.exit(1)
    print(f"\nOK: no category dropped more than {args.threshold:.2f}")


if __name__ == "__main__":
    main()
