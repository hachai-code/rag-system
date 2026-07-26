"""LLM-as-judge for the innerdance RAG answers.

One narrow judge per dimension, never a mega-prompt: each of the rubric dimensions
(A-F in rubrics.md) is scored by its own narrow call, so a single dimension's PASS/FAIL
can't bleed into another and we can track per-dimension accuracy. Each verdict is a
PASS/FAIL plus a one-sentence rationale — the rationale is the whole point, it's how
you debug a judge that disagrees with you.

Judge model is DeepSeek V4 Flash, chosen for cost. Known trade-off: it also writes
the answers (self-agreement bias) and is a weak judge — the per-verdict rationales
and the human labels (judge_vs_human.py) are the check on it. The judged answers come
from the live RAG, so this also exercises retrieval + generation end to end against
the human eval set.

Structured output via instructor + Pydantic (instructor.Mode.TOOLS uses Anthropic's
tool-calling to fill the model), so a verdict either parses into `Verdict` or retries.

Runs are resumable: each row is appended and flushed, keyed by eval-item id.

Run: uv run python -m evals.answer.judge [n]
"""

import json
import os
import sys
from pathlib import Path

import instructor
import psycopg
from pydantic import BaseModel, Field

from evals.schema import load_jsonl
from rag import answer, search
from rag.clients import OPENROUTER_BASE_URL
from rag.config import CONFIG
from rag.db import connect
from rag.query.retrieve import NO_ANSWER, no_relevant_hits

EVAL_FILE = Path(__file__).parent / "data" / "rag_system_human_eval.jsonl"
OUT_FILE = Path(__file__).parent / "data" / "judgments.jsonl"
JUDGE_MODEL = CONFIG.gen_models["flash"]  # cost-driven; validated against human labels
# $/token by model id (OpenRouter list prices). Cost math keys off JUDGE_MODEL so a
# judge swap reprices automatically — or KeyErrors loudly instead of lying silently.
PRICES = {"deepseek/deepseek-v4-flash": (0.09 / 1_000_000, 0.18 / 1_000_000)}
IN_PRICE, OUT_PRICE = PRICES[JUDGE_MODEL]
# Flash is a reasoning model; left on, it can spend the whole max_tokens budget
# thinking and return an empty verdict (IncompleteOutputException). A one-sentence
# PASS/FAIL doesn't need chain-of-thought. OpenRouter unified param:
# https://openrouter.ai/docs/use-cases/reasoning-tokens
REASONING_OFF = {"reasoning": {"effort": "none"}}
# Sentinel for a hard model refusal (stop_reason="refusal"): the API returns no text,
# so there is nothing to judge against the rubric. Items 70-F/74-E hit this — we record
# the refusal as the answer so the row produces a verdict instead of crashing.
REFUSED = "[hard refusal: the model declined to answer]"


def judge_client(provider: str, model: str):
    """Build the instructor judge client for the configured provider: Anthropic native,
    or an OpenAI-compatible model (e.g. DeepSeek V4 Flash via OpenRouter). Both fill the
    Verdict via structured output, so callers don't care which they got."""
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


class Verdict(BaseModel):
    # rationale before passed so the model reasons before it concludes — the verdict
    # reads off the rationale rather than the other way round.
    rationale: str = Field(description="One sentence naming the specific reason for the verdict.")
    passed: bool = Field(description="True if the answer meets this dimension's PASS criteria.")


# The taxonomy (codes, names, criteria, severity) has one home: evals/taxonomy.json.
# rubrics.md is the human-readable prose; the labelling UI legend reads the same JSON.
TAXONOMY = json.loads((Path(__file__).parents[1] / "taxonomy.json").read_text())
RUBRICS = {code: (t["name"], t["criterion"], t["pass"], t["fail"]) for code, t in TAXONOMY.items()}

SYSTEM = """You are a strict evaluator for a RAG system that answers questions over the 'innerdance' corpus (a body-based consciousness practice plus a neuroscience book it draws on).

Judge ONE dimension only — {name} — and ignore every other dimension.

Criterion: {criterion}
PASS: {pass_def}
FAIL: {fail_def}

Judge only against the criterion above. A coherent, readable answer is not "garbled" or "incoherent" — do not invent a fluency objection that the criterion doesn't name. FAIL only when the criterion's FAIL condition is actually met.

Give a one-sentence rationale naming the specific reason, then decide PASS (passed=true) or FAIL (passed=false)."""

# A hard refusal carries no answer to grade, so the rubric can't be applied. Record a
# fixed verdict rather than calling the model: the refusal itself is the finding.
REFUSAL_VERDICT = Verdict(
    rationale="The model hard-refused (stop_reason='refusal'); there is no answer to grade against this dimension.",
    passed=False,
)


def judge_with_usage(client, code: str, question: str, answer_text: str, ideal: str = ""):
    """One narrow judge call for a rubric dimension. Returns (Verdict, in_tokens, out_tokens);
    usage comes via create_with_completion because plain create() drops the raw response."""
    if answer_text == REFUSED:
        return REFUSAL_VERDICT, 0, 0
    name, criterion, pass_def, fail_def = RUBRICS[code]
    user = f"Question:\n{question}\n\nAnswer to judge:\n{answer_text}"
    if ideal:
        user += f"\n\nReference answer (ground truth):\n{ideal}"
    verdict, completion = client.create_with_completion(
        max_tokens=1000,
        max_retries=2,  # instructor re-asks the model if the output doesn't parse to Verdict
        response_model=Verdict,
        extra_body=REASONING_OFF,
        messages=[
            {
                "role": "system",
                "content": SYSTEM.format(
                    name=name, criterion=criterion, pass_def=pass_def, fail_def=fail_def
                ),
            },
            {"role": "user", "content": user},
        ],
    )
    # Anthropic usage exposes input_tokens/output_tokens; OpenAI-compatible (OpenRouter)
    # exposes prompt_tokens/completion_tokens — accept either so one helper serves both.
    usage = completion.usage
    in_tok = getattr(usage, "input_tokens", None) or usage.prompt_tokens
    out_tok = getattr(usage, "output_tokens", None) or usage.completion_tokens
    return verdict, in_tok, out_tok


def judge_dimension(client, code: str, question: str, answer_text: str, ideal: str = "") -> Verdict:
    return judge_with_usage(client, code, question, answer_text, ideal)[0]


def rag_answer(conn: psycopg.Connection, question: str) -> str:
    """Answer exactly as app.py's /ask does: search, the relevance gate, then generate.
    A hard model refusal (stop_reason='refusal') has no text to return, so we surface
    the REFUSED sentinel instead of letting the empty content crash the caller."""
    hits = search(conn, question)
    if no_relevant_hits(hits):
        return NO_ANSWER
    text, _ = answer(question, hits)
    return text or REFUSED


def eval_items() -> list[dict]:
    return load_jsonl(EVAL_FILE)


def existing_ids() -> set[int]:
    return {row["id"] for row in load_jsonl(OUT_FILE)}


def main() -> None:
    items = eval_items()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(items)
    client = judge_client("openai-compat", JUDGE_MODEL)
    done = existing_ids()

    with connect() as conn, OUT_FILE.open("a") as out:
        judged = 0
        for item in items:
            if judged >= n:
                break
            if item["id"] in done:
                continue
            try:
                ans = rag_answer(conn, item["question"])
                verdicts = {
                    code: judge_dimension(
                        client, code, item["question"], ans, item["ideal_answer"]
                    ).model_dump()
                    for code in item["axial_codes"]
                }
            except Exception as e:
                # One bad item (network, parse failure past retries) shouldn't abort the run.
                print(f"  [skip] item {item['id']}: {type(e).__name__}: {e}")
                continue
            row = {
                "id": item["id"],
                "question": item["question"],
                "difficulty": item["difficulty"],
                "answer": ans,
                "verdicts": verdicts,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            judged += 1
            marks = " ".join(f"{c}:{'P' if v['passed'] else 'F'}" for c, v in verdicts.items())
            print(f"  [{judged:>2}/{n}] item {item['id']:>2}  {marks}  {item['question'][:44]}")

    print(f"\n{OUT_FILE} judged {judged} new item(s)")


if __name__ == "__main__":
    main()
