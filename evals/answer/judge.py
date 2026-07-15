"""LLM-as-judge for the innerdance RAG answers.

One narrow judge per dimension, never a mega-prompt: each of the rubric dimensions
(A-E in rubrics.md) is scored by its own Opus call, so a single dimension's PASS/FAIL
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

from rag import answer, search
from rag.clients import OPENROUTER_BASE_URL
from rag.config import CONFIG
from rag.db import connect
from rag.query.retrieve import RELEVANCE_THRESHOLD

EVAL_FILE = Path(__file__).parent / "data" / "rag_system_human_eval.jsonl"
OUT_FILE = Path(__file__).parent / "data" / "judgments.jsonl"
JUDGE_MODEL = CONFIG.gen_models["flash"]  # cost-driven; validated against human labels
# Flash is a reasoning model; left on, it can spend the whole max_tokens budget
# thinking and return an empty verdict (IncompleteOutputException). A one-sentence
# PASS/FAIL doesn't need chain-of-thought. OpenRouter unified param:
# https://openrouter.ai/docs/use-cases/reasoning-tokens
REASONING_OFF = {"reasoning": {"effort": "none"}}
NO_ANSWER = "I don't have information on that in the innerdance corpus."
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
        "Is the answer grounded in context that is on-topic for the question asked, drawn from the right speaker's turn, and complete enough to answer fully?",
        "Grounded in on-topic context from the correct speaker/turn, with no obviously missing key information.",
        "Draws on adjacent-but-wrong-topic material, uses the wrong speaker, or is incomplete because context is missing.",
    ),
    "C": (
        "Generation Quality",
        "Does the response genuinely synthesize the material into a coherent answer at the depth the question deserves, rather than parroting the corpus, mishandling the task type, or under-generating?",
        "Reasons over and integrates the content into a clear, original synthesis at appropriate depth; handles creative/'make a content piece' requests generatively while staying grounded.",
        "Repeats corpus text near-verbatim without understanding, is disjointed/confusing, treats a generative request as a flat lookup, or is over-generalized/cut short as if truncated (under-generation).",
    ),
    "D": (
        "Grounding, Provenance & Factual Validation",
        "Are factual claims — especially neuroscience/physiology claims — grounded in the corpus or a vetted source, or appropriately hedged rather than stated as unverified fact? And does the answer distinguish the innerdance framework's own teaching from the supplementary neuroscience book it draws on?",
        "Falsifiable claims are grounded or clearly framed as the innerdance teaching rather than established science; book-derived material is attributed to the book, not presented as framework teaching; adds helpful external context where useful.",
        "States unverified neuro/medical claims as settled fact, presents the supplementary book's material as innerdance framework teaching (or vice versa), or omits helpful external context where the corpus alone is insufficient.",
    ),
    "E": (
        "Formatting & Conventions",
        "Does the response follow surface conventions — citation markers where claims are made, and correct terminology?",
        "Citations present where claims are made; terminology/style followed (e.g. 'innerdance' lowercase, joined).",
        "Missing citation markers, or violates terminology/style rules.",
    ),
    "F": (
        "Corpus & Data Quality",
        "Does the answer reflect clean underlying data — correcting known transcription errors (it's KAP, not 'CAP') and keeping speaker turns straight rather than crediting one speaker's words to another?",
        "Uses corrected terminology (KAP) and attributes statements to the right speaker/turn; does not propagate a transcription artifact.",
        "Repeats a transcription error (e.g. 'CAP' for KAP) as if correct, or conflates/mis-attributes speaker turns (a student's line credited to the teacher).",
    ),
}

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


def judge_dimension(client, code: str, question: str, answer_text: str, ideal: str = "") -> Verdict:
    if answer_text == REFUSED:
        return REFUSAL_VERDICT
    name, criterion, pass_def, fail_def = RUBRICS[code]
    user = f"Question:\n{question}\n\nAnswer to judge:\n{answer_text}"
    if ideal:
        user += f"\n\nReference answer (ground truth):\n{ideal}"
    return client.create(
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


def rag_answer(conn: psycopg.Connection, question: str) -> str:
    """Answer exactly as app.py's /ask does: search, the relevance gate, then generate.
    A hard model refusal (stop_reason='refusal') has no text to return, so we surface
    the REFUSED sentinel instead of letting the empty content crash the caller."""
    hits = search(conn, question)
    if not hits or hits[0]["distance"] > RELEVANCE_THRESHOLD:
        return NO_ANSWER
    text, _ = answer(question, hits)
    return text or REFUSED


def eval_items() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


def existing_ids() -> set[int]:
    if not OUT_FILE.exists():
        return set()
    return {json.loads(line)["id"] for line in OUT_FILE.read_text().splitlines() if line.strip()}


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
