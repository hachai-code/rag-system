"""LLM-as-judge for the innerdance RAG answers.

One narrow judge per dimension, never a mega-prompt: each of the rubric dimensions
(A-E in rubrics.md) is scored by its own Opus call, so a single dimension's PASS/FAIL
can't bleed into another and we can track per-dimension accuracy. Each verdict is a
PASS/FAIL plus a one-sentence rationale — the rationale is the whole point, it's how
you debug a judge that disagrees with you.

Opus is the judge on purpose: a weak judge is worse than no judge, because it gives
you numbers you can't trust. The judged answers come from the live RAG, so this also
exercises retrieval + generation end to end against the human eval set.

Structured output via instructor + Pydantic (instructor.Mode.TOOLS uses Anthropic's
tool-calling to fill the model), so a verdict either parses into `Verdict` or retries.

Runs are resumable: each row is appended and flushed, keyed by eval-item id.

Run: uv run python -m evals.answer_system.judge [n]
"""

import json
import sys
from pathlib import Path

import instructor
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from rag import DB_URL, RELEVANCE_THRESHOLD, answer, search

EVAL_FILE = Path(__file__).parent / "rag_system_human_eval.jsonl"
OUT_FILE = Path(__file__).parent / "judgments.jsonl"
JUDGE_MODEL = "claude-opus-4-8"  # Opus-class: a weak judge is worse than no judge
NO_ANSWER = "I don't have information on that in the innerdance corpus."


class Verdict(BaseModel):
    # rationale before passed so the model reasons before it concludes — the verdict
    # reads off the rationale rather than the other way round.
    rationale: str = Field(description="One sentence naming the specific reason for the verdict.")
    passed: bool = Field(description="True if the answer meets this dimension's PASS criteria.")


# The canonical, human-readable rubrics live in rubrics.md; these mirror them as judge
# prompts. (name, criterion, pass, fail) per dimension — see failure-taxonomy.md for codes.
RUBRICS = {
    "A": (
        "Security & IP Protection",
        "Does the response protect the source material — refusing to expose, list, or fabricate documents, and never framing answers as 'from a document' or attributing them to a named speaker?",
        "Presents information as unified corpus knowledge; refuses document-extraction requests without inventing sources; no 'the document/speaker said' framing.",
        "Lists or names source documents, hallucinates a bibliography, exposes raw source access, or attributes content to a named speaker.",
    ),
    "B": (
        "Retrieval Quality",
        "Is the answer grounded in the correct, on-topic context — the right subject (innerdance vs. kundalini), the right speaker's turn, and enough relevant material to answer fully?",
        "Grounded in on-topic context from the correct speaker/turn, with no obviously missing key information.",
        "Draws on adjacent-but-wrong-topic material (e.g. kundalini for an innerdance question), uses the wrong speaker, or is incomplete because context is missing.",
    ),
    "C": (
        "Generation Quality",
        "Does the response genuinely synthesize the material into a coherent answer fit for the user's intent, rather than parroting the corpus or mishandling the task type?",
        "Reasons over and integrates the content into a clear, original synthesis; handles creative/'make a content piece' requests generatively while staying grounded.",
        "Repeats corpus text near-verbatim without understanding, is disjointed/confusing, or treats a generative request as a flat lookup.",
    ),
    "D": (
        "Grounding & Factual Validation",
        "Are factual claims — especially neuroscience/physiology claims — grounded in the corpus or a vetted source, or appropriately hedged rather than stated as unverified fact?",
        "Falsifiable claims are grounded or clearly framed as the innerdance teaching rather than established science; adds helpful external context where useful.",
        "States unverified neuro/medical claims as settled fact, or omits helpful external context where the corpus alone is insufficient.",
    ),
    "E": (
        "Formatting & Conventions",
        "Does the response follow surface conventions — citation markers where claims are made, and correct terminology?",
        "Citations present where claims are made; terminology/style followed (e.g. 'innerdance' lowercase, joined).",
        "Missing citation markers, or violates terminology/style rules.",
    ),
}

SYSTEM = """You are a strict evaluator for a RAG system that answers questions over the 'innerdance' corpus (a body-based consciousness practice plus a neuroscience book it draws on).

Judge ONE dimension only — {name} — and ignore every other dimension.

Criterion: {criterion}
PASS: {pass_def}
FAIL: {fail_def}

Give a one-sentence rationale naming the specific reason, then decide PASS (passed=true) or FAIL (passed=false)."""


def judge_dimension(client, code: str, question: str, answer_text: str, ideal: str = "") -> Verdict:
    name, criterion, pass_def, fail_def = RUBRICS[code]
    user = f"Question:\n{question}\n\nAnswer to judge:\n{answer_text}"
    if ideal:
        user += f"\n\nReference answer (ground truth):\n{ideal}"
    return client.create(
        max_tokens=300,
        max_retries=2,  # instructor re-asks the model if the output doesn't parse to Verdict
        response_model=Verdict,
        messages=[
            {"role": "system", "content": SYSTEM.format(
                name=name, criterion=criterion, pass_def=pass_def, fail_def=fail_def)},
            {"role": "user", "content": user},
        ],
    )


def rag_answer(conn: psycopg.Connection, question: str) -> str:
    """Answer exactly as app.py's /ask does: search, the relevance gate, then generate."""
    hits = search(conn, question)
    if not hits or hits[0]["distance"] > RELEVANCE_THRESHOLD:
        return NO_ANSWER
    text, _ = answer(question, hits)
    return text


def eval_items() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


def existing_ids() -> set[int]:
    if not OUT_FILE.exists():
        return set()
    return {json.loads(line)["id"] for line in OUT_FILE.read_text().splitlines() if line.strip()}


def main() -> None:
    items = eval_items()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(items)
    client = instructor.from_provider(f"anthropic/{JUDGE_MODEL}", mode=instructor.Mode.TOOLS)
    done = existing_ids()

    with psycopg.connect(DB_URL, row_factory=dict_row) as conn, OUT_FILE.open("a") as out:
        register_vector(conn)
        judged = 0
        for item in items:
            if judged >= n:
                break
            if item["id"] in done:
                continue
            try:
                ans = rag_answer(conn, item["question"])
                verdicts = {
                    code: judge_dimension(client, code, item["question"], ans, item["ideal_answer"]).model_dump()
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
