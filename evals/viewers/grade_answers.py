"""Render an HTML grading sheet for answer_feedback.jsonl.

For each item it shows the question, the reference answer and the RAG answer
side by side, whether the app retrieved the gold chunk, and up/down buttons. Grades
persist in the browser (localStorage), and a "Download graded JSONL" button writes
them back into the rows' `grade` field so you end up with the graded document.

Reads answer_feedback.jsonl; open grade_answers.html in a browser afterwards.

Run: uv run evals/viewers/grade_answers.py
"""

import html
import json
from pathlib import Path

IN_FILE = Path(__file__).parent.parent / "answer" / "data" / "answer_feedback.jsonl"
OUT_FILE = Path(__file__).parent / "grade_answers.html"

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
.src { color: #888; font-weight: 400; font-size: .85rem; }
.gold, .miss { font-size: .72rem; padding: .1rem .45rem; border-radius: 20px; font-weight: 600; }
.gold { background: #e6f6e6; color: #295; } .miss { background: #fce8e8; color: #c33; }
.question { font-size: 1.15rem; font-weight: 600; margin: .4rem 0 .8rem; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.col { padding: .7rem; border-radius: 6px; }
.col h3 { margin: 0 0 .4rem; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
.ref { background: #f3faf4; } .rag { background: #f4f7fb; }
.grades { display: flex; gap: .5rem; }
.g { padding: .25rem .7rem; border: 1px solid #ccc; border-radius: 20px; cursor: pointer; }
.g input { margin-right: .3rem; }
.g-up { border-color: #3a3; } .g-down { border-color: #d44; }
"""

JS = """
function tally() {
  const counts = {up:0, down:0, ungraded:0};
  const cards = document.querySelectorAll('.card');
  const state = {};
  cards.forEach(card => {
    const sel = card.querySelector('input[type=radio]:checked');
    if (sel) { counts[sel.value]++; state[sel.name] = sel.value; }
    else counts.ungraded++;
  });
  document.getElementById('tally').textContent =
    `up ${counts.up}   down ${counts.down}   ·   ${counts.ungraded}/${cards.length} ungraded`;
  localStorage.setItem('answer_grades', JSON.stringify(state));
}
function restore() {
  const state = JSON.parse(localStorage.getItem('answer_grades') || '{}');
  for (const [name, val] of Object.entries(state)) {
    const el = document.querySelector(`input[name="${name}"][value="${val}"]`);
    if (el) el.checked = true;
  }
  tally();
}
function reset() { localStorage.removeItem('answer_grades');
  document.querySelectorAll('input[type=radio]:checked').forEach(r => r.checked = false); tally(); }
function download() {
  const grades = JSON.parse(localStorage.getItem('answer_grades') || '{}');
  const lines = ROWS.map(r => JSON.stringify({...r, grade: grades['grade-' + r.id] || ''}));
  const blob = new Blob([lines.join('\\n') + '\\n'], {type: 'application/jsonl'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'answer_feedback.graded.jsonl'; a.click();
}
document.addEventListener('change', e => { if (e.target.matches('input[type=radio]')) tally(); });
window.addEventListener('load', restore);
"""

GRADES = ["up", "down"]


def br(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_card(row: dict) -> str:
    radios = "".join(
        f'<label class="g g-{g}"><input type="radio" name="grade-{row["id"]}" value="{g}">{g}</label>'
        for g in GRADES
    )
    badge = '<span class="gold">gold retrieved</span>' if row["retrieved_gold"] \
        else '<span class="miss">gold missed</span>'
    return f"""
<section class="card">
  <div class="qhead">
    <h2>Q{row['id']} <span class="src">{html.escape(row['source']['title'])}</span> {badge}</h2>
    <div class="grades">{radios}</div>
  </div>
  <p class="question">{html.escape(row['question'])}</p>
  <div class="cols">
    <div class="col ref"><h3>Reference</h3><div>{br(row['reference_answer'])}</div></div>
    <div class="col rag"><h3>RAG answer</h3><div>{br(row['rag_answer'])}</div></div>
  </div>
</section>"""


def main() -> None:
    rows = [json.loads(line) for line in IN_FILE.read_text().splitlines() if line.strip()]
    cards = "".join(render_card(row) for row in rows)
    header = (
        '<div id="bar"><span id="tally"></span>'
        '<button onclick="download()">download graded jsonl</button>'
        '<button onclick="reset()">reset</button></div>'
        "<h1>RAG answer grading</h1>"
        "<p>Grade each <b>RAG answer</b> against the reference. Grades save in your browser; "
        "use <b>download</b> to write them back into the rows.</p>"
    )
    data = f"<script>const ROWS = {json.dumps(rows, ensure_ascii=False)};</script>"
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Answer grading</title><style>{CSS}</style></head><body>"
        f"{header}{cards}{data}<script>{JS}</script></body></html>"
    )
    OUT_FILE.write_text(page)
    print(f"Wrote {OUT_FILE} ({len(rows)} items). Open it in a browser.")


if __name__ == "__main__":
    main()
