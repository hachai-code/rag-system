"""Build the seed file for the labelling app.

Joins three sources from the rag-system repo by question id:
  - questions      <- rag_system_human_eval.jsonl
  - rag answers    <- judgments.jsonl (field "answer", same ids)
  - existing labels<- rag_system_human_eval.txt (free-text Label: blocks)

Writes labelling_seed.jsonl: {id, question, rag_answer, label} per line.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "answer" / "data"
EVAL = BASE / "rag_system_human_eval.jsonl"
JUDG = BASE / "judgments.jsonl"
TXT = BASE / "rag_system_human_eval.txt"
OUT = Path(__file__).parent / "labelling_seed.jsonl"


def norm(q):
    # alphanumerics only, so spacing/punctuation drift between .txt and .jsonl
    # (e.g. "arising(maybe" vs "arising (maybe") still matches.
    return re.sub(r"[^a-z0-9]", "", q.lower())


def parse_txt_labels(text):
    """Map normalized question -> combined label text from the freeform .txt.

    Blocks are blank-line separated; a block's first line is its header
    (Question:/Label:/Ideal answer:). A Label attaches to the most recent
    Question; orphan labels before any question are ignored.
    """
    labels = {}
    current = None
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip("\n").split("\n")
        header = lines[0].strip().rstrip(":").lower()
        content = "\n".join(lines[1:]).strip()
        if header.startswith("question"):
            current = norm(content)
        elif header.startswith("label") and current:
            labels[current] = f"{labels[current]}\n{content}".strip() if current in labels else content
    return labels


def main():
    eval_rows = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    answers = {r["id"]: r["answer"]
               for r in (json.loads(l) for l in JUDG.read_text().splitlines() if l.strip())}
    # Stop before the taxonomy write-up so it isn't parsed as labels.
    labels = parse_txt_labels(TXT.read_text().split("-----")[0])

    rows = [{
        "id": r["id"],
        "question": r["question"],
        "rag_answer": answers.get(r["id"], ""),
        "label": labels.get(norm(r["question"]), ""),
    } for r in eval_rows]

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))

    labelled = sum(1 for r in rows if r["label"])
    no_answer = sum(1 for r in rows if not r["rag_answer"])
    used = {norm(r["question"]) for r in eval_rows} & set(labels)
    print(f"wrote {len(rows)} -> {OUT.name}")
    print(f"  {labelled} have an existing label, {no_answer} missing rag_answer")
    print(f"  {len(labels)} label blocks in .txt, {len(used)} matched a question")
    unmatched = set(labels) - used
    if unmatched:
        print("  UNMATCHED labels (question text differs between .txt and .jsonl):")
        for q in unmatched:
            print(f"    - {q[:80]}")


if __name__ == "__main__":
    main()
