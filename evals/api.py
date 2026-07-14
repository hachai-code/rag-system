"""Read-only eval-results API for the dashboard (frontend/app/evals/page.tsx).

One endpoint, one payload: per-run summaries (chronological) for the trend charts,
plus a dev-vs-test comparison for the "final" config — the config hash of the most
recent run. Split isn't persisted in eval_results, so it's joined here from the
eval set by question_id.

Kept import-light on purpose: no evals.run / evals.answer.judge (their import chain
pulls the judge client into the serving process). No rate limit — the limiter lives
in rag/app.py and this is a read-only local aggregate.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from rag.db import connect

EVAL_FILE = Path(__file__).parent / "answer" / "data" / "rag_system_human_eval.jsonl"

router = APIRouter()


class RunSummary(BaseModel):
    run_id: int
    created_at: datetime
    git_sha: str
    config_name: str | None
    config_hash: str | None
    n: int
    pass_rate: dict[str, float]
    cost: float
    latency_ms: float


class SplitSummary(BaseModel):
    run_id: int
    n: int
    pass_rate: dict[str, float]


class FinalConfig(BaseModel):
    config_name: str | None
    config_hash: str | None
    dev: SplitSummary | None
    test: SplitSummary | None


class EvalsSummary(BaseModel):
    runs: list[RunSummary]
    final: FinalConfig | None


def _split_map() -> dict[int, str]:
    return {
        row["id"]: row["split"]
        for row in map(json.loads, EVAL_FILE.read_text().splitlines())
    }


def _pass_rate(rows: list[dict]) -> dict[str, float]:
    per_dim = defaultdict(list)
    for r in rows:
        for dim, passed in r["scores"].items():
            per_dim[dim].append(passed)
    return {dim: sum(v) / len(v) for dim, v in sorted(per_dim.items())}


@router.get("/evals/summary")
def evals_summary() -> EvalsSummary:
    with connect() as conn:
        rows = conn.execute(
            """SELECT r.id AS run_id, r.created_at, r.git_sha,
                      r.config->>'name' AS config_name, r.config->>'hash' AS config_hash,
                      e.question_id, e.scores, e.cost, e.latency_ms
               FROM eval_runs r JOIN eval_results e ON e.run_id = r.id
               ORDER BY r.created_at, r.id"""
        ).fetchall()
    if not rows:
        return EvalsSummary(runs=[], final=None)

    by_run: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)

    runs = [
        RunSummary(
            run_id=run_id,
            created_at=rs[0]["created_at"],
            git_sha=rs[0]["git_sha"],
            config_name=rs[0]["config_name"],
            config_hash=rs[0]["config_hash"],
            n=len(rs),
            pass_rate=_pass_rate(rs),
            cost=float(sum(r["cost"] for r in rs)),
            latency_ms=sum(r["latency_ms"] for r in rs) / len(rs),
        )
        for run_id, rs in by_run.items()
    ]

    # Final config = the most recent run's hash (fallback: name, fallback: the run
    # itself). Per split, the newest matching run with judged items in that split —
    # covers both separate --split dev/test runs and one --split all run.
    last = runs[-1]
    matching = [
        rs
        for rs in reversed(list(by_run.values()))
        if (rs[0]["config_hash"] or rs[0]["config_name"] or rs[0]["run_id"])
        == (last.config_hash or last.config_name or last.run_id)
    ]
    splits = _split_map()
    final = FinalConfig(config_name=last.config_name, config_hash=last.config_hash,
                        dev=None, test=None)
    for split in ("dev", "test"):
        for rs in matching:
            in_split = [r for r in rs if splits.get(r["question_id"]) == split]
            if in_split:
                setattr(final, split, SplitSummary(
                    run_id=rs[0]["run_id"], n=len(in_split),
                    pass_rate=_pass_rate(in_split),
                ))
                break

    return EvalsSummary(runs=runs, final=final)
