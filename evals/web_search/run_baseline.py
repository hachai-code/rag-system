"""Run the web search agent once over eval_set.jsonl and append raw results to
results.jsonl for hand-labeling. Resumable: already-answered ids are skipped, so
a crashed run doesn't re-pay for finished questions.

Run from the repo root:  uv run python -m evals.web_search.run_baseline
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

from langfuse import get_client

import rag.query.web_search_agent as agent
from rag.config import CONFIG

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"

# Baseline runs on the cheap model; recorded in every row.
agent.MODEL = CONFIG.gen_models["flash"]


def langfuse_stats(question: str) -> dict:
    """Pull the run's span metadata (iterations, cost, limit_hit, ...) from
    Langfuse. Best effort — returns {} rather than ever failing the run."""
    try:
        host = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
        auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
        for _ in range(3):  # ingestion lags a few seconds behind flush()
            time.sleep(5)
            traces = httpx.get(f"{host}/api/public/traces",
                               params={"name": "web-search-agent", "limit": 1},
                               auth=auth, timeout=30).json()["data"]
            if not traces or traces[0].get("input") != question:
                continue
            obs = httpx.get(f"{host}/api/public/observations",
                            params={"traceId": traces[0]["id"], "limit": 100},
                            auth=auth, timeout=30).json()["data"]
            root = next(o for o in obs
                        if o["type"] == "SPAN" and not o.get("parentObservationId"))
            meta = root.get("metadata") or {}
            return {"trace_id": traces[0]["id"],
                    **{k: meta.get(k) for k in
                       ("iterations", "cost_usd", "limit_hit", "citation_retries")}}
        return {}
    except Exception:
        return {}


def main(ids: set[int] | None = None, tag: str = "") -> None:
    out = Path(__file__).parent / (f"results_{tag}.jsonl" if tag else "results.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open() if line.strip()}

    for row in (json.loads(line) for line in EVAL_SET.open()):
        if ids is not None and row["id"] not in ids:
            continue
        if row["id"] in done:
            print(f"[{row['id']:2}] already in results — skipping")
            continue
        print(f"[{row['id']:2}] {row['question'][:75]}")
        start = time.monotonic()
        answer = agent.run_agent(row["question"])
        seconds = round(time.monotonic() - start, 1)
        get_client().flush()
        result = {
            **row,
            "answer": answer,
            "model": agent.MODEL,
            "seconds": seconds,
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **langfuse_stats(row["question"]),
        }
        with out.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"     done in {seconds}s\n")

    print(f"all requested questions present in {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="", help="comma-separated question ids (default: all)")
    parser.add_argument("--tag", default="", help="write results_<tag>.jsonl instead of results.jsonl")
    args = parser.parse_args()
    main({int(i) for i in args.ids.split(",") if i} or None, args.tag)
