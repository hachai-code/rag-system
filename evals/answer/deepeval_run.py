"""DeepEval RAG-triad metrics over the innerdance eval set, judged by DeepSeek V4 Pro.

Additive to the A-F rubric judges (answer/judge.py): those stay binary and
Anthropic-Opus-judged; this adds the standard continuous RAG-triad scores DeepEval
gives off the shelf — faithfulness, answer relevancy, and contextual
relevancy/precision/recall — each a 0-1 score + a one-line reason.

Answers come from the live RAG (search -> relevance gate -> generate), exactly as
app.py's /ask, so a run exercises retrieval + generation end to end. Every eval item
has an ideal_answer, so the two reference-based metrics (precision, recall) run on all.

The judge is DeepSeek V4 Pro. DeepEval's built-in DeepSeekModel only knows model IDs
up to v3.x, so we wrap DeepSeek's OpenAI-compatible endpoint with instructor (already
a project dep) — schema-validated output with retries, and it sidesteps that stale
model-ID table.

Runs are resumable: each row is appended and flushed, keyed by eval-item id.

Run:   uv run python -m evals.answer.deepeval_run [n]      # judge first n items (default all)
Smoke: uv run python -m evals.answer.deepeval_run smoke    # one fixed case, assert wiring works
"""

import json
import os
import sys
from pathlib import Path

import instructor
from openai import OpenAI

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from evals.answer.judge import NO_ANSWER
from rag import RELEVANCE_THRESHOLD, answer, search
from rag.db import connect

EVAL_FILE = Path(__file__).parent.parent / "eval_set.jsonl"
OUT_FILE = Path(__file__).parent / "data" / "deepeval_results.jsonl"
JUDGE_MODEL = "deepseek-v4-pro"


class DeepSeekJudge(DeepEvalBaseLLM):
    """DeepSeek V4 Pro as a DeepEval judge, via its OpenAI-compatible endpoint + instructor."""

    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        super().__init__(model)

    def load_model(self):
        client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
        return instructor.from_openai(client, mode=instructor.Mode.JSON)

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.model.chat.completions.create(
                model=self.model_name,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
            )
        completion = self.model.chat.completions.create(
            model=self.model_name,
            response_model=None,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self.model_name


def metrics(judge: DeepSeekJudge) -> dict:
    """The RAG triad, keyed by the label used in the output rows. async_mode=False so each
    measure() runs through our synchronous judge rather than DeepEval's own event loop."""
    kw = {"model": judge, "async_mode": False}
    return {
        "contextual_relevancy": ContextualRelevancyMetric(**kw),
        "faithfulness": FaithfulnessMetric(**kw),
        "answer_relevancy": AnswerRelevancyMetric(**kw),
        "contextual_precision": ContextualPrecisionMetric(**kw),
        "contextual_recall": ContextualRecallMetric(**kw),
    }


def rag_answer(conn, question: str) -> tuple[str, list[dict]]:
    """Answer as /ask does — search, relevance gate, generate — but keep the hits so they
    can become the DeepEval retrieval_context."""
    hits = search(conn, question)
    if not hits or hits[0]["distance"] > RELEVANCE_THRESHOLD:
        return NO_ANSWER, hits
    text, _ = answer(question, hits)
    return text, hits


def score(metric, tc: LLMTestCase) -> dict:
    try:
        metric.measure(tc)
        return {"score": metric.score, "success": metric.is_successful(), "reason": metric.reason}
    except Exception as e:
        # One metric's failure (e.g. empty retrieval_context on a gated answer) shouldn't
        # drop the other four for this item.
        return {"error": f"{type(e).__name__}: {e}"}


def eval_items() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


def existing_ids() -> set:
    if not OUT_FILE.exists():
        return set()
    return {json.loads(line)["id"] for line in OUT_FILE.read_text().splitlines() if line.strip()}


def main() -> None:
    items = eval_items()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(items)
    judge = DeepSeekJudge()
    mets = metrics(judge)
    done = existing_ids()

    with connect() as conn, OUT_FILE.open("a") as out:
        judged = 0
        for item in items:
            if judged >= n:
                break
            if item["id"] in done:
                continue
            ans, hits = rag_answer(conn, item["question"])
            tc = LLMTestCase(
                input=item["question"],
                actual_output=ans,
                expected_output=item["ideal_answer"],
                retrieval_context=[h["content"] for h in hits],
            )
            results = {name: score(m, tc) for name, m in mets.items()}
            row = {"id": item["id"], "question": item["question"], "answer": ans, "metrics": results}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            judged += 1
            marks = " ".join(f"{k[:4]}:{v.get('score', 'err')}" for k, v in results.items())
            print(f"  [{judged:>2}/{n}] item {item['id']:>2}  {marks}  {item['question'][:40]}")

    print(f"\n{OUT_FILE} judged {judged} new item(s)")


def smoke() -> None:
    """One fixed case through all five metrics — confirms the DeepSeek wiring before a real run."""
    tc = LLMTestCase(
        input="What is innerdance?",
        actual_output="Innerdance is an experiential somatic practice that guides people into "
        "altered states through breath, music, and movement [1].",
        expected_output="Innerdance is a facilitated somatic/energetic practice using breath, "
        "sound and movement to reach non-ordinary states.",
        retrieval_context=[
            "Innerdance is a practice where participants lie down and let the body move "
            "spontaneously, guided by music and breath into altered states of consciousness.",
        ],
    )
    for name, m in metrics(DeepSeekJudge()).items():
        r = score(m, tc)
        assert "error" not in r, f"{name} failed: {r['error']}"
        assert 0.0 <= r["score"] <= 1.0, f"{name} score out of range: {r['score']}"
        assert r["reason"], f"{name} returned empty reason"
        print(f"  ok  {name:<22} score={r['score']:.2f}  {r['reason'][:60]}")
    print("smoke passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        main()
