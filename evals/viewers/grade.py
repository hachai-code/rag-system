"""Render an HTML grading sheet for the eval questions.

For each question it shows the question, the RAG answer and the ideal answer
side by side, the retrieved chunks, and radio buttons to grade the RAG answer
(correct / partial / wrong / hallucinated) with a live tally. Grades persist in
the browser (localStorage); "download" writes them back into the rows' `grade`
field so they can leave the browser.

Reuses the stored `draft_answer` (the RAG answer from gen_eval.py) and re-runs
search() to show the chunks it was based on — retrieval is deterministic, so the
chunks match the answer. Open evals/viewers/grade.html in a browser afterwards.

Run: uv run python -m evals.viewers.grade
"""

import html
from pathlib import Path

from evals.schema import load_jsonl
from evals.viewers._sheet import br, grade_radios, render_grading_sheet
from rag import search
from rag.db import connect

EVAL_FILE = Path(__file__).parent.parent / "eval_set.jsonl"
OUT_FILE = Path(__file__).parent / "grade.html"
GRADES = ["correct", "partial", "wrong", "hallucinated"]
TALLY_LABELS = {"correct": "✅", "partial": "\U0001f7e1", "wrong": "❌", "hallucinated": "\U0001f47b"}

CSS = """
.cat { color: #888; font-weight: 400; font-size: .85rem; }
.rag { background: #f4f7fb; } .ideal { background: #f3faf4; }
.g-correct { border-color: #3a3; } .g-partial { border-color: #db3; }
.g-wrong { border-color: #d44; } .g-hallucinated { border-color: #a3a; }
details { margin-top: .8rem; } summary { cursor: pointer; color: #555; }
.chunk { border-left: 3px solid #ccd; padding: .3rem .6rem; margin: .5rem 0; font-size: .85rem; }
.dist { color: #c33; font-weight: 600; } .ctitle { color: #666; }
.ctext { color: #444; margin-top: .2rem; white-space: pre-wrap; }
"""


def render_chunks(hits: list[dict]) -> str:
    out = []
    for h in hits:
        body = html.escape(h["content"][:700]) + ("…" if len(h["content"]) > 700 else "")
        out.append(
            f'<div class="chunk"><span class="dist">{h["distance"]:.3f}</span> '
            f'<span class="ctitle">{html.escape(h["title"][:55])}</span>'
            f'<div class="ctext">{body}</div></div>'
        )
    return "\n".join(out)


def render_card(row: dict, hits: list[dict]) -> str:
    return f"""
<section class="card">
  <div class="qhead">
    <h2>Q{row["id"]} <span class="cat">{row["category"]}</span></h2>
    <div class="grades">{grade_radios(row["id"], GRADES)}</div>
  </div>
  <p class="question">{html.escape(row["question"])}</p>
  <div class="cols">
    <div class="col rag"><h3>RAG answer</h3><div>{br(row["draft_answer"])}</div></div>
    <div class="col ideal"><h3>Ideal answer (draft — adjust)</h3><div>{br(row["ideal_answer"])}</div></div>
  </div>
  <details><summary>Retrieved chunks (top {len(hits)})</summary>{render_chunks(hits)}</details>
</section>"""


def main() -> None:
    rows = load_jsonl(EVAL_FILE)
    cards = []
    with connect() as conn:
        for row in rows:
            hits = search(conn, row["question"])
            cards.append(render_card(row, hits))
            print(f"  rendered Q{row['id']}", flush=True)

    page = render_grading_sheet(
        rows=rows,
        cards="".join(cards),
        grades=GRADES,
        storage_key="rag_grades",
        download_name="eval_set.graded.jsonl",
        title="RAG answer grading",
        intro="Grade each <b>RAG answer</b> against the ideal. Grades save in your browser; "
        "use <b>download</b> to write them back into the rows.",
        extra_css=CSS,
        tally_labels=TALLY_LABELS,
    )
    OUT_FILE.write_text(page)
    print(f"\nWrote {OUT_FILE} ({len(rows)} questions). Open it in a browser.")


if __name__ == "__main__":
    main()
