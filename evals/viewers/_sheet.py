"""Shared core for the static HTML grading sheets (grade.py, grade_answers.py).

One page skeleton, one localStorage persistence + tally + reset script, one
download-back-to-JSONL path. Each viewer supplies its rows, its card body, and its
grade vocabulary; grades live under a per-viewer localStorage key so two sheets
don't clobber each other."""

import html
import json

BASE_CSS = """
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 1rem; color: #222; }
#bar { position: sticky; top: 0; background: #fff; border-bottom: 2px solid #ddd;
       padding: .6rem 0; margin-bottom: 1rem; font-size: 1.1rem; z-index: 10; }
#tally { font-weight: 600; }
button { font: inherit; padding: .2rem .6rem; margin-left: 1rem; cursor: pointer; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.qhead { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.qhead h2 { margin: 0; font-size: 1.05rem; }
.question { font-size: 1.15rem; font-weight: 600; margin: .4rem 0 .8rem; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.col { padding: .7rem; border-radius: 6px; }
.col h3 { margin: 0 0 .4rem; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }
.grades { display: flex; gap: .5rem; flex-wrap: wrap; }
.g { padding: .25rem .6rem; border: 1px solid #ccc; border-radius: 20px; cursor: pointer; font-size: .85rem; }
.g input { margin-right: .25rem; }
"""

# Placeholders (%GRADES% etc.) are injected in render_grading_sheet, same idiom as
# trace_viewer's %THRESHOLD%.
JS = """
const GRADES = %GRADES%;
const LABELS = %LABELS%;
const KEY = %KEY%;
const ROWS = %ROWS%;
function tally() {
  const counts = Object.fromEntries(GRADES.map(g => [g, 0]));
  let ungraded = 0;
  const cards = document.querySelectorAll('.card');
  const state = {};
  cards.forEach(card => {
    const sel = card.querySelector('input[type=radio]:checked');
    if (sel) { counts[sel.value]++; state[sel.name] = sel.value; }
    else ungraded++;
  });
  document.getElementById('tally').textContent =
    GRADES.map(g => `${LABELS[g] || g} ${counts[g]}`).join('   ') +
    `   ·   ${ungraded}/${cards.length} ungraded`;
  localStorage.setItem(KEY, JSON.stringify(state));
}
function restore() {
  const state = JSON.parse(localStorage.getItem(KEY) || '{}');
  for (const [name, val] of Object.entries(state)) {
    const el = document.querySelector(`input[name="${name}"][value="${val}"]`);
    if (el) el.checked = true;
  }
  tally();
}
function reset() { localStorage.removeItem(KEY);
  document.querySelectorAll('input[type=radio]:checked').forEach(r => r.checked = false); tally(); }
function download() {
  const grades = JSON.parse(localStorage.getItem(KEY) || '{}');
  const lines = ROWS.map(r => JSON.stringify({...r, grade: grades['grade-' + r.id] || ''}));
  const blob = new Blob([lines.join('\\n') + '\\n'], {type: 'application/jsonl'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = %DOWNLOAD_NAME%; a.click();
}
document.addEventListener('change', e => { if (e.target.matches('input[type=radio]')) tally(); });
window.addEventListener('load', restore);
"""


def br(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def grade_radios(row_id, grades: list[str]) -> str:
    return "".join(
        f'<label class="g g-{g}"><input type="radio" name="grade-{row_id}" value="{g}">{g}</label>'
        for g in grades
    )


def render_grading_sheet(
    *,
    rows: list[dict],
    cards: str,
    grades: list[str],
    storage_key: str,
    download_name: str,
    title: str,
    intro: str,
    extra_css: str = "",
    tally_labels: dict[str, str] | None = None,
) -> str:
    """Assemble the full grading page: sticky tally/download/reset bar, the viewer's
    cards, and the persistence script. `rows` are embedded so download writes grades
    back into the original records."""
    js = (
        JS.replace("%GRADES%", json.dumps(grades))
        .replace("%LABELS%", json.dumps(tally_labels or {}))
        .replace("%KEY%", json.dumps(storage_key))
        .replace("%ROWS%", json.dumps(rows, ensure_ascii=False))
        .replace("%DOWNLOAD_NAME%", json.dumps(download_name))
    )
    header = (
        '<div id="bar"><span id="tally"></span>'
        '<button onclick="download()">download graded jsonl</button>'
        '<button onclick="reset()">reset</button></div>'
        f"<h1>{title}</h1><p>{intro}</p>"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{BASE_CSS}{extra_css}</style></head><body>"
        f"{header}{cards}<script>{js}</script></body></html>"
    )
