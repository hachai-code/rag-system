"""Render an HTML grading sheet for answer_feedback.jsonl.

For each item it shows the question, the reference answer and the RAG answer
side by side, whether the app retrieved the gold chunk, and up/down buttons. Grades
persist in the browser (localStorage), and a "Download graded JSONL" button writes
them back into the rows' `grade` field so you end up with the graded document.

Reads answer_feedback.jsonl; open grade_answers.html in a browser afterwards.

Run: uv run evals/viewers/grade_answers.py
"""

import html
from pathlib import Path

from evals.schema import load_jsonl
from evals.viewers._sheet import br, grade_radios, render_grading_sheet

IN_FILE = Path(__file__).parent.parent / "answer" / "data" / "answer_feedback.jsonl"
OUT_FILE = Path(__file__).parent / "grade_answers.html"

GRADES = ["up", "down"]

CSS = """
.src { color: #888; font-weight: 400; font-size: .85rem; }
.gold, .miss { font-size: .72rem; padding: .1rem .45rem; border-radius: 20px; font-weight: 600; }
.gold { background: #e6f6e6; color: #295; } .miss { background: #fce8e8; color: #c33; }
.ref { background: #f3faf4; } .rag { background: #f4f7fb; }
.g-up { border-color: #3a3; } .g-down { border-color: #d44; }
"""


def render_card(row: dict) -> str:
    badge = (
        '<span class="gold">gold retrieved</span>'
        if row["retrieved_gold"]
        else '<span class="miss">gold missed</span>'
    )
    return f"""
<section class="card">
  <div class="qhead">
    <h2>Q{row["id"]} <span class="src">{html.escape(row["source"]["title"])}</span> {badge}</h2>
    <div class="grades">{grade_radios(row["id"], GRADES)}</div>
  </div>
  <p class="question">{html.escape(row["question"])}</p>
  <div class="cols">
    <div class="col ref"><h3>Reference</h3><div>{br(row["reference_answer"])}</div></div>
    <div class="col rag"><h3>RAG answer</h3><div>{br(row["rag_answer"])}</div></div>
  </div>
</section>"""


def main() -> None:
    rows = load_jsonl(IN_FILE)
    page = render_grading_sheet(
        rows=rows,
        cards="".join(render_card(row) for row in rows),
        grades=GRADES,
        storage_key="answer_grades",
        download_name="answer_feedback.graded.jsonl",
        title="RAG answer grading",
        intro="Grade each <b>RAG answer</b> against the reference. Grades save in your "
        "browser; use <b>download</b> to write them back into the rows.",
        extra_css=CSS,
    )
    OUT_FILE.write_text(page)
    print(f"Wrote {OUT_FILE} ({len(rows)} items). Open it in a browser.")


if __name__ == "__main__":
    main()
