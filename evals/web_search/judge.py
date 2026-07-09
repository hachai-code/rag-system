"""LLM-as-judge for the web-search agent baseline.

Two narrow judges per result row, never a mega-prompt (the evals/answer/judge.py
convention): stage 1 grades the answer against the eval set's must_contain +
facts, quoting verbatim evidence per fact so every verdict is auditable; stage 2
runs only on non-correct rows and assigns failure categories from a fixed
taxonomy, with open coding for failure mechanisms the taxonomy doesn't capture.
Stage 2 sees the run's trajectory (search queries, fetched URLs, tool errors)
pulled from Langfuse, because bad queries and premature stops are trajectory
properties, not answer properties.

Judge model is deepseek-v4-flash — the same model that wrote the answers, chosen
for cost. Known trade-off: self-agreement bias and a weak judge; the evidence
quotes and the user's hand labels are the check on it.

Runs are resumable: judgments append one flushed row per eval id.

Run: uv run python -m evals.web_search.judge
"""

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parents[2] / ".env")

from rag.config import CONFIG
from rag.query.web_search_agent import _cited_urls

JUDGE_MODEL = CONFIG.gen_models["flash"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
METRICS = Path(__file__).parent / "analysis" / "baseline_metrics.json"
NO_ANSWER = "Could not produce an answer."

Category = Literal["bad_search_queries", "wrong_source_selection", "synthesis_error",
                   "premature_stop", "citation_mismatch", "other"]


class FactVerdict(BaseModel):
    fact: str
    evidence: str = Field(description="Verbatim quote from the answer supporting the "
                                      "verdict; empty string if the fact is absent.")
    status: Literal["present", "missing", "contradicted"]


class Correctness(BaseModel):
    # rationale first so the verdict reads off the reasoning, not the other way round
    rationale: str = Field(description="One or two sentences naming the specific reason.")
    fact_verdicts: list[FactVerdict]
    overall: Literal["correct", "partial", "wrong"]


class FailureAnalysis(BaseModel):
    rationale: str = Field(description="One or two sentences naming the failure mechanism.")
    categories: list[Category] = Field(description="All that apply, primary cause first.")
    emerging_category: str = Field(default="", description="Short kebab-case name for a "
        "failure mechanism the fixed categories don't capture; empty if they do.")
    emerging_definition: str = Field(default="", description="One sentence defining the "
        "emerging category; empty if none.")


CORRECTNESS_SYSTEM = """You are a strict evaluator grading a web-research agent's answer against ground-truth criteria.

For each ground-truth fact, decide its status in the answer:
- "present" — stated correctly; put the verbatim supporting sentence from the answer in evidence.
- "missing" — not addressed; evidence is an empty string.
- "contradicted" — the answer asserts something incompatible with the fact; quote the offending text.

Then decide the overall verdict by applying these rules mechanically — they override your own sense of what a good answer is:
- "correct": the must-contain criterion is satisfied AND every single fact has status "present". If even one fact is "missing", the verdict cannot be "correct".
- "wrong": the must-contain criterion is unmet, OR any fact is "contradicted", OR the text does not actually answer the question.
- "partial": everything else — a real answer with some facts missing.

Judge only against the criteria. Do not penalize style, length, extra detail, or an early-stop note. Give the rationale first."""

FAILURE_SYSTEM = """You are diagnosing WHY a web-research agent's answer fell short. You get the question, per-fact verdicts, the answer, and (when available) the run's trajectory: the search queries it issued, the pages it fetched, tool errors, and whether a budget limit ended the run.

Assign every category that applies, primary cause first:
- bad_search_queries — queries too vague, wrong terms, or wrong year, so the right sources never surfaced.
- wrong_source_selection — the right results surfaced, but the agent fetched or relied on wrong or low-quality pages.
- synthesis_error — the needed facts were in the trajectory, but were combined, calculated, or stated wrongly.
- premature_stop — a budget limit ended the run, or the agent stopped researching before gathering what was needed (includes fallback answers that fail).
- citation_mismatch — claims attributed to sources that don't support them, or citation-rejection loops.
- other — anything else.

Prefer these fixed categories. But if the actual failure mechanism is not genuinely captured by them — or filing it under "other" would hide a describable, recurring-looking pattern — name an emerging category (short kebab-case) and define it in one sentence. Ground everything in the trajectory when you have it; give the rationale first."""


def _langfuse_get(path: str, **params) -> list[dict]:
    host = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    return httpx.get(f"{host}/api/public/{path}", params=params, auth=auth,
                     timeout=30).json()["data"]


def trajectory_digest(row: dict) -> str:
    """Ordered tool calls + run stats for the row's trace. Best effort — returns ''
    rather than ever failing the judge run."""
    try:
        trace_id = row.get("trace_id")
        if not trace_id:
            traces = _langfuse_get("traces", name="web-search-agent", limit=50)
            trace_id = next(t["id"] for t in traces if t.get("input") == row["question"])
        obs = _langfuse_get("observations", traceId=trace_id, limit=100)
        lines = []
        for o in sorted(obs, key=lambda o: o["startTime"]):
            if o["type"] == "TOOL":
                arg = o.get("input")
                arg = json.dumps(arg, ensure_ascii=False) if isinstance(arg, dict) else str(arg)
                out = str(o.get("output") or "").replace("\n", " ")
                lines.append(f"{o['name']}({arg}) -> {out[:120]}")
            elif o["type"] == "EVENT" and o["name"] == "citation-rejected":
                lines.append(f"citation-rejected: {o.get('input')}")
        stats = {k: row.get(k) for k in ("iterations", "limit_hit", "seconds")}
        return f"run stats: {stats}\n" + "\n".join(lines)
    except Exception:
        return ""


def judge_correctness(client, row: dict) -> Correctness:
    if row["answer"].startswith(NO_ANSWER):
        return Correctness(
            rationale="The agent produced no answer (best-effort fallback failed).",
            fact_verdicts=[FactVerdict(fact=f, evidence="", status="missing")
                           for f in row["facts"]],
            overall="wrong",
        )
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(row["facts"], 1))
    return client.chat.completions.create(
        model=JUDGE_MODEL, max_tokens=2000, max_retries=2, response_model=Correctness,
        messages=[
            {"role": "system", "content": CORRECTNESS_SYSTEM},
            {"role": "user", "content": f"Question:\n{row['question']}\n\n"
                                        f"A correct answer must contain:\n{row['must_contain']}\n\n"
                                        f"Ground-truth facts:\n{facts}\n\n"
                                        f"Answer to grade:\n{row['answer']}"},
        ],
    )


def judge_failure(client, row: dict, verdict: Correctness) -> FailureAnalysis:
    digest = trajectory_digest(row)
    trajectory = digest if digest else "(trajectory unavailable — judge from the answer alone)"
    verdicts = "\n".join(f"- [{v.status}] {v.fact}" for v in verdict.fact_verdicts)
    return client.chat.completions.create(
        model=JUDGE_MODEL, max_tokens=1500, max_retries=2, response_model=FailureAnalysis,
        messages=[
            {"role": "system", "content": FAILURE_SYSTEM},
            {"role": "user", "content": f"Question:\n{row['question']}\n\n"
                                        f"Verdict: {verdict.overall} — {verdict.rationale}\n"
                                        f"Fact verdicts:\n{verdicts}\n\n"
                                        f"Trajectory:\n{trajectory}\n\n"
                                        f"Answer:\n{row['answer']}"},
        ],
    )


def write_metrics(rows: list[dict], judgments: dict[int, dict], tag: str = "") -> None:
    overall = Counter(j["overall"] for j in judgments.values())
    facts_total = sum(len(r["facts"]) for r in rows)
    facts_present = sum(1 for j in judgments.values()
                        for v in j["fact_verdicts"] if v["status"] == "present")
    per_category = {}
    for r in rows:
        c = per_category.setdefault(r["category"], Counter())
        c[judgments[r["id"]]["overall"]] += 1
    failures = Counter()
    emerging = Counter()
    for j in judgments.values():
        if j.get("failure"):
            failures.update(j["failure"]["categories"])
            label = j["failure"]["emerging_category"].strip().lower().replace(" ", "-")
            if label:
                emerging[label] += 1
    # Gate metric: judged correct AND carries at least one (mechanically
    # trace-verified) markdown citation link.
    correct_and_cited = sum(
        1 for r in rows
        if judgments[r["id"]]["overall"] == "correct" and _cited_urls(r["answer"])
    )
    metrics = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_model": rows[0]["model"],
        "judge_model": JUDGE_MODEL,
        "n": len(rows),
        "overall": dict(overall),
        "correct_and_cited": correct_and_cited,
        "fact_recall": f"{facts_present}/{facts_total}",
        "per_category": {k: dict(v) for k, v in sorted(per_category.items())},
        "failure_categories": dict(failures.most_common()),
        "emerging_categories": dict(emerging.most_common()),
    }
    # Debug-round/gate runs get their own metrics file; the baseline reference
    # is only ever written by an untagged full run.
    out = Path(__file__).parent / "analysis" / f"metrics_{tag}.json" if tag else METRICS
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    n_correct = overall.get("correct", 0)
    fail_str = ", ".join(f"{k} x{v}" for k, v in failures.most_common()) or "none"
    print(f"\n{n_correct}/{len(rows)} correct ({overall.get('partial', 0)} partial, "
          f"{overall.get('wrong', 0)} wrong) | correct-and-cited {correct_and_cited}/{len(rows)} "
          f"| fact recall {facts_present}/{facts_total} | failures: {fail_str}")
    for label, n in emerging.most_common():
        flag = "  << taxonomy-v2 candidate" if n >= 2 else ""
        print(f"  emerging: {label} x{n}{flag}")
    print(f"wrote {out}")


def main(tag: str = "") -> None:
    data = Path(__file__).parent / "data"
    results = data / (f"results_{tag}.jsonl" if tag else "results.jsonl")
    judgments_path = data / (f"judgments_{tag}.jsonl" if tag else "judgments.jsonl")
    client = instructor.from_openai(
        OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"]),
        mode=instructor.Mode.TOOLS,
    )
    rows = [json.loads(line) for line in results.open()]
    judged = {}
    if judgments_path.exists():
        judged = {j["id"]: j for j in map(json.loads, judgments_path.open()) if j}

    for row in rows:
        if row["id"] in judged:
            print(f"[{row['id']:2}] already judged — skipping")
            continue
        verdict = judge_correctness(client, row)
        judgment = {
            "id": row["id"],
            "category": row["category"],
            "overall": verdict.overall,
            "rationale": verdict.rationale,
            "fact_verdicts": [v.model_dump() for v in verdict.fact_verdicts],
            "failure": None,
            "judge_model": JUDGE_MODEL,
        }
        if verdict.overall != "correct":
            judgment["failure"] = judge_failure(client, row, verdict).model_dump()
        with judgments_path.open("a") as f:
            f.write(json.dumps(judgment, ensure_ascii=False) + "\n")
        judged[row["id"]] = judgment
        fail = (judgment["failure"] or {}).get("categories", [])
        print(f"[{row['id']:2}] {verdict.overall:<8} {fail if fail else ''}")

    write_metrics(rows, judged, tag)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="",
                        help="read results_<tag>.jsonl, write judgments_<tag>.jsonl, skip metrics file")
    main(parser.parse_args().tag)
