"""Inspect AI port of a 15-question eval subset (learning day — the hand-rolled
harness in evals/run.py stays canonical).

    uv run inspect eval evals/inspect-ai/task.py --model openrouter/deepseek/deepseek-v4-flash

See NOTES.md alongside this file for what Inspect standardizes vs. what run.py does
that Inspect doesn't.
"""

# ruff: noqa: E402 — the sys.path insert must precede the rag/evals imports
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # inspect loads this file by path; rag/evals imports need repo root

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from evals.answer.judge import NO_ANSWER, REFUSED, RUBRICS, SYSTEM
from rag import answer, retrieve
from rag.db import connect
from rag.query.retrieve import covered

# Stratified 15/75 mirroring the axial-code distribution, all dev split.
# C=5 (20*,25,26,38,45) D=3 (3*,12,68) A=2 (51,55) B=2 (59+,61) E=1 (73) F=1 (69) multi=1 (23: C,D)
# * = has ideal_answer; + = expected to trip the covered() no-answer gate
SUBSET = {3, 12, 20, 23, 25, 26, 38, 45, 51, 55, 59, 61, 68, 69, 73}

# Mirrors evals/configs/baseline.json (evals are flash-only; answer() defaults to v4-pro).
GEN_MODEL = "deepseek/deepseek-v4-flash"
TOP_K, THRESHOLD, METHOD, FMT = 5, 0.7, "rerank", "claims"

EVAL_FILE = ROOT / "evals/answer/data/rag_system_human_eval.jsonl"


@solver
def rag_pipeline():
    """Answer with the real production pipeline (covered -> retrieve -> answer), not
    Inspect's generate(): the eval subject is the system, prompt plumbing included.
    Trade-off: the generation call is opaque to Inspect (no token counts/caching)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        def run(q: str):
            with connect() as conn:
                ok, _ = covered(conn, q, THRESHOLD)
                if not ok:
                    return NO_ANSWER, []
                hits = retrieve(conn, q, k=TOP_K, method=METHOD)
                text, _ = answer(q, hits, model=GEN_MODEL, fmt=FMT)
                return text or REFUSED, hits

        # the pipeline is sync (psycopg/instructor); thread it so samples parallelize
        text, hits = await asyncio.to_thread(run, state.input_text)
        state.metadata["retrieved"] = [f"{h['title']}: {h['content'][:200]}" for h in hits]
        state.output = ModelOutput.from_content(model="rag-pipeline", content=text)
        return state

    return solve


@scorer(metrics={"*": [mean(), stderr()]})
def rubric_judge():
    """One narrow judge call per axial code (same design as evals/answer/judge.py),
    emitted as a dict-valued Score so Inspect aggregates a metric column per code."""

    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model()
        ans = state.output.completion
        # Inspect requires identical keys across samples' dict scores; NaN = "not
        # applicable" and is excluded from the per-code mean/stderr aggregation.
        values: dict[str, float] = {code: float("nan") for code in RUBRICS}
        notes = []
        for code in state.metadata["axial_codes"]:
            if ans == REFUSED:
                values[code], text = 0, "hard refusal; nothing to grade"
            else:
                name, criterion, pass_def, fail_def = RUBRICS[code]
                prompt = SYSTEM.format(
                    name=name, criterion=criterion, pass_def=pass_def, fail_def=fail_def
                )
                prompt += f"\n\nQuestion:\n{state.input_text}\n\nAnswer to judge:\n{ans}"
                if target.text:
                    prompt += f"\n\nReference answer (ground truth):\n{target.text}"
                prompt += '\n\nEnd with one final line: "VERDICT: PASS" or "VERDICT: FAIL".'
                result = await grader.generate(prompt)
                text = result.completion
                m = re.search(r"VERDICT:\s*(PASS|FAIL)", text)
                values[code] = 1 if m and m.group(1) == "PASS" else 0
            notes.append(f"[{code}] {text.strip()}")
        # safety-critical rule from rubrics.md: any dimension FAIL fails the trace
        values["trace"] = float(all(values[c] == 1 for c in state.metadata["axial_codes"]))
        return Score(value=values, answer=ans[:300], explanation="\n\n".join(notes))

    return score


def record_to_sample(r: dict) -> Sample:
    return Sample(
        input=r["question"],
        target=r.get("ideal_answer") or "",
        id=r["id"],
        metadata={"axial_codes": r["axial_codes"], "difficulty": r["difficulty"]},
    )


@task
def rag_rubric_eval() -> Task:
    ds = json_dataset(str(EVAL_FILE), record_to_sample)
    return Task(
        dataset=ds.filter(lambda s: s.id in SUBSET),
        solver=rag_pipeline(),
        scorer=rubric_judge(),
    )
