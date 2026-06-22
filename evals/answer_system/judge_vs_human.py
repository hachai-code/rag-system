"""Agreement between the LLM judge and the human labels in rag_system_human_eval.txt.

The .txt labels are free-text notes, so they're hand-coded here into (eval id,
dimension, PASS/FAIL) verdicts — only the labels that express a clear verdict on a
dimension the judge actually scores. Each tuple cites the original note.

Caveat baked into the number: the human labelled answers seen at review time; the
judge grades the RAG's answers *today*. Where the system now behaves differently
(e.g. the doc-extraction attack now refuses instead of hallucinating), a mismatch is
the system changing, not the judge erring. Treat kappa as a noisy floor, n is tiny.

Run: uv run python -m evals.answer_system.judge_vs_human
"""

import json
from pathlib import Path

JUDGMENTS = Path(__file__).parent / "judgments.jsonl"

# (eval_id, dimension, human_pass, source note from the .txt)
HUMAN = [
    (2,  "B", False, "'life force energy' attributed to wrong speaker — needs speaker distinction"),
    (11, "D", False, "'Pull from external sources to have additional helpful information'"),
    (20, "C", True,  "'Answered really well by finding relevant context for the question.'"),
    (29, "C", True,  "'Very good answer'"),
    (30, "C", True,  "'Answer pretty good'"),
    (31, "C", True,  "'Answer really good'"),
    (32, "C", False, "treated a creative task as a lookup — 'should be more independently creative'"),
    (33, "A", False, "hallucinated bibliography from a book that doesn't exist — 'the answer is wrong'"),
    (34, "B", False, "answered with kundalini pricing, not innerdance"),
    (36, "A", False, "'a speaker is mentioned, but the rag system should give objective information'"),
    (43, "C", True,  "'Good answer solving the paradox by calling for intuition'"),
    (44, "E", False, "'innerdance is written like this with small caps and connected'"),
    (47, "C", False, "'doesn't create proper synthesis, but just repeats what was written'"),
]


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Two-rater, two-category (PASS/FAIL) kappa."""
    n = len(pairs)
    po = sum(h == j for h, j in pairs) / n
    h_pass = sum(h for h, _ in pairs) / n
    j_pass = sum(j for _, j in pairs) / n
    pe = h_pass * j_pass + (1 - h_pass) * (1 - j_pass)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> None:
    verdicts = {}  # (id, dim) -> judge passed
    for line in JUDGMENTS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for dim, v in row["verdicts"].items():
            verdicts[(row["id"], dim)] = v["passed"]

    pairs, missing = [], []
    print(f"{'id':>3} {'dim':>3} {'human':>6} {'judge':>6}  note")
    for eid, dim, human_pass, note in HUMAN:
        judge_pass = verdicts.get((eid, dim))
        if judge_pass is None:
            missing.append((eid, dim))
            continue
        pairs.append((human_pass, judge_pass))
        mark = "ok " if human_pass == judge_pass else "DIFF"
        print(f"{eid:>3} {dim:>3} {'PASS' if human_pass else 'FAIL':>6} "
              f"{'PASS' if judge_pass else 'FAIL':>6}  {mark} {note}")

    n = len(pairs)
    agree = sum(h == j for h, j in pairs)
    print(f"\nn = {n} labelled (id, dimension) pairs"
          + (f"  ·  missing from judgments: {missing}" if missing else ""))
    print(f"agreement = {agree}/{n} = {agree / n:.0%}")
    print(f"cohen's kappa = {cohens_kappa(pairs):.2f}")


if __name__ == "__main__":
    main()
