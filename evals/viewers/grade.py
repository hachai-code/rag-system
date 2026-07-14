"""Render an HTML grading sheet for the eval questions.

For each question it shows the question, the RAG answer and the ideal answer
side by side, the retrieved chunks, and radio buttons to grade the RAG answer
(correct / partial / wrong / hallucinated) with a live tally. Grades persist in
the browser (localStorage) so you can grade in one sitting without losing work.

Reuses the stored `draft_answer` (the RAG answer from gen_eval.py) and re-runs
search() to show the chunks it was based on — retrieval is deterministic, so the
chunks match the answer. Open evals/grade.html in a browser afterwards.

Run: uv run python -m evals.viewers.grade
"""

import html
import json
from pathlib import Path

from rag import search
from rag.db import connect

EVAL_FILE = Path(__file__).parent.parent / "eval_set.jsonl"
OUT_FILE = Path(__file__).parent / "grade.html"
GRADES = ["correct", "partial", "wrong", "hallucinated"]

CSS = """
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 1rem; color: #222; }
#bar { position: sticky; top: 0; background: #fff; border-bottom: 2px solid #ddd;
       padding: .6rem 0; margin-bottom: 1rem; font-size: 1.1rem; z-index: 10; }
#tally { font-weight: 600; }
button { font: inherit; padding: .2rem .6rem; margin-left: 1rem; cursor: pointer; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.qhead { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.qhead h2 { margin: 0; font-size: 1.05rem; }
.cat { color: #888; font-weight: 400; font-size: .85rem; }
.question { font-size: 1.15rem; font-weight: 600; margin: .4rem 0 .8rem; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.col { padding: .7rem; border-radius: 6px; }
.col h3 { margin: 0 0 .4rem; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
.rag { background: #f4f7fb; } .ideal { background: #f3faf4; }
.grades { display: flex; gap: .5rem; flex-wrap: wrap; }
.g { padding: .25rem .6rem; border: 1px solid #ccc; border-radius: 20px; cursor: pointer; font-size: .85rem; }
.g input { margin-right: .25rem; }
.g-correct { border-color: #3a3; } .g-partial { border-color: #db3; }
.g-wrong { border-color: #d44; } .g-hallucinated { border-color: #a3a; }
details { margin-top: .8rem; } summary { cursor: pointer; color: #555; }
.chunk { border-left: 3px solid #ccd; padding: .3rem .6rem; margin: .5rem 0; font-size: .85rem; }
.dist { color: #c33; font-weight: 600; } .ctitle { color: #666; }
.ctext { color: #444; margin-top: .2rem; white-space: pre-wrap; }
"""

JS = """
function tally() {
  const counts = {correct:0, partial:0, wrong:0, hallucinated:0, ungraded:0};
  const cards = document.querySelectorAll('.card');
  const state = {};
  cards.forEach(card => {
    const sel = card.querySelector('input[type=radio]:checked');
    if (sel) { counts[sel.value]++; state[sel.name] = sel.value; }
    else counts.ungraded++;
  });
  document.getElementById('tally').textContent =
    `✅ ${counts.correct}   \U0001F7E1 ${counts.partial}   ❌ ${counts.wrong}` +
    `   \U0001F47B ${counts.hallucinated}   ·   ${counts.ungraded}/${cards.length} ungraded`;
  localStorage.setItem('rag_grades', JSON.stringify(state));
}
function restore() {
  const state = JSON.parse(localStorage.getItem('rag_grades') || '{}');
  for (const [name, val] of Object.entries(state)) {
    const el = document.querySelector(`input[name="${name}"][value="${val}"]`);
    if (el) el.checked = true;
  }
  tally();
}
function reset() { localStorage.removeItem('rag_grades');
  document.querySelectorAll('input[type=radio]:checked').forEach(r => r.checked = false); tally(); }
document.addEventListener('change', e => { if (e.target.matches('input[type=radio]')) tally(); });
window.addEventListener('load', restore);
"""


def br(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


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
    radios = "".join(
        f'<label class="g g-{g}"><input type="radio" name="grade-{row["id"]}" value="{g}">{g}</label>'
        for g in GRADES
    )
    return f"""
<section class="card">
  <div class="qhead">
    <h2>Q{row['id']} <span class="cat">{row['category']}</span></h2>
    <div class="grades">{radios}</div>
  </div>
  <p class="question">{html.escape(row['question'])}</p>
  <div class="cols">
    <div class="col rag"><h3>RAG answer</h3><div>{br(row['draft_answer'])}</div></div>
    <div class="col ideal"><h3>Ideal answer (draft — adjust)</h3><div>{br(row['ideal_answer'])}</div></div>
  </div>
  <details><summary>Retrieved chunks (top {len(hits)})</summary>{render_chunks(hits)}</details>
</section>"""


def main() -> None:
    rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    cards = []
    with connect() as conn:
        for row in rows:
            hits = search(conn, row["question"])
            cards.append(render_card(row, hits))
            print(f"  rendered Q{row['id']}", flush=True)

    header = (
        '<div id="bar"><span id="tally"></span>'
        '<button onclick="reset()">reset</button></div>'
        "<h1>RAG answer grading</h1>"
        "<p>Grade each <b>RAG answer</b> against the ideal. Grades save in your browser.</p>"
    )
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>RAG grading</title><style>{CSS}</style></head><body>"
        f"{header}{''.join(cards)}<script>{JS}</script></body></html>"
    )
    OUT_FILE.write_text(page)
    print(f"\nWrote {OUT_FILE} ({len(rows)} questions). Open it in a browser.")


if __name__ == "__main__":
    main()
