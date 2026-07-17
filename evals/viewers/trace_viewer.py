"""Trace viewer for open coding — review real RAG traces and write free-text notes.

Open coding (see the Pragmatic Engineer evals piece) is the first qualitative pass:
read whole traces and jot what you *notice*, before inventing any error categories.
So this tool shows the full trace — question, the retrieved chunks, the answer — and
gives one free-text box per trace. There are no predefined labels on purpose; the
categories come *later*, by clustering these notes (that's how failure-taxonomy.md
was built).

A trace is pulled straight from the live RAG: search() for the chunks, answer() for
the generation (or the NO_ANSWER relevance gate). Chunks are captured at generation
time, so the retrieval you review is exactly what produced the answer — that's what
lets you spot the *first upstream failure* (bad retrieval vs. good chunks generated
badly). eval_results doesn't store chunks, which is why we pull live rather than read
the table.

    uv run python -m evals.viewers.trace_viewer pull   # build traces.jsonl from the eval set
    uv run python -m evals.viewers.trace_viewer        # serve the viewer (default), port 5003
"""

import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from rag.db import connect
from rag.query.gate import ask_gate
from rag.query.retrieve import RELEVANCE_THRESHOLD

HERE = Path(__file__).parent
QUESTIONS = HERE.parent / "answer" / "data" / "rag_system_human_eval.jsonl"
TRACES = HERE / "traces.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pull_trace(conn, question: str) -> dict:
    """One trace from the live pipeline, via the shared ask_gate: the chunks that produced
    the answer plus the answer itself (or the refusal the relevance gate returns, so a
    refused query is reviewable too). Rerank-fused hits can carry no vector distance."""
    result = ask_gate(conn, question)
    chunks = [
        {
            "title": h["title"],
            "source": h["source"],
            "distance": round(float(h["distance"]), 4) if h.get("distance") is not None else None,
            "content": h["content"],
        }
        for h in result.hits
    ]
    return {"chunks": chunks, "answer": result.answer}


def pull() -> None:
    """Generate a trace per eval question. Resumable: skips ids already in traces.jsonl."""
    done = {t["id"] for t in load(TRACES)}
    todo = [q for q in load(QUESTIONS) if q["id"] not in done]
    if not todo:
        print(f"{TRACES}: already complete ({len(done)} traces)")
        return
    with connect() as conn, TRACES.open("a") as out:
        for q in todo:
            trace = pull_trace(conn, q["question"])
            out.write(
                json.dumps(
                    {"id": q["id"], "question": q["question"], **trace, "note": ""},
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            print(f"  pulled #{q['id']:>2}  {len(trace['chunks'])} chunks  {q['question'][:50]}")
    print(f"{TRACES}: {len(load(TRACES))} traces")


app = FastAPI()


@app.get("/api/traces")
def api_traces() -> dict:
    return {"source": QUESTIONS.name, "traces": load(TRACES)}


@app.post("/api/note/{tid}")
async def api_note(tid: int, req: Request) -> dict:
    note = (await req.json()).get("note", "")
    traces = load(TRACES)
    for t in traces:
        if t["id"] == tid:
            t["note"] = note
            break
    TRACES.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces))
    return {"ok": True}


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Trace viewer — open coding</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body { max-width: 1100px; margin: 2rem auto; padding: 0 1rem;
         font: 16px/1.5 system-ui, sans-serif; color: #1a1a1a; }
  h1 { font-size: 1.5rem; }
  a { cursor: pointer; color: #2962ff; }
  button { font: inherit; padding: 0.4rem 0.9rem; cursor: pointer; }
  .toolbar { display: flex; gap: 1rem; align-items: center; margin: 1rem 0; }
  .muted { color: #888; font-size: 0.85rem; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem; margin: 1rem 0; }
  .q { font-weight: 600; margin: 0.25rem 0 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  label { display: block; font-weight: 600; margin: 0.5rem 0 0.25rem; }
  textarea { width: 100%; font: inherit; padding: 0.5rem; box-sizing: border-box; height: 16rem; }
  .status { color: green; font-size: 0.85rem; }
  .answer { max-height: 30rem; overflow: auto; line-height: 1.6; background: #fff;
            padding: 0.75rem 1rem; border: 1px solid #ccc; border-radius: 6px; }
  details { margin-top: 0.75rem; }
  summary { cursor: pointer; font-weight: 600; }
  .chunk { border-left: 3px solid #ccc; padding: 0.25rem 0.75rem; margin: 0.5rem 0; }
  .chunk .meta { font-size: 0.8rem; color: #555; }
  .chunk .text { white-space: pre-wrap; max-height: 14rem; overflow: auto; font-size: 0.9rem; }
  .gated { color: #b00; font-size: 0.8rem; }
</style>
<div id="app"></div>
<script>
const THRESHOLD = %THRESHOLD%;

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    else if (k === "style") el.style.cssText = v;
    else if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el[k] = v;
    else el.setAttribute(k, v);
  }
  for (const kid of kids) el.append(kid?.nodeType ? kid : document.createTextNode(kid ?? ""));
  return el;
}

// Debounced autosave: POST the note ~0.5s after typing pauses (and on blur), update
// only the status line so the cursor never jumps.
function autosave(textarea, id, status) {
  let timer;
  const fire = () => fetch(`/api/note/${id}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note: textarea.value }),
    }).then(() => status.textContent = "saved ✓ " + new Date().toLocaleTimeString());
  textarea.oninput = () => { clearTimeout(timer); timer = setTimeout(fire, 500); };
  textarea.onblur = fire;
}

function chunkEl(c) {
  return h("div", { class: "chunk" },
    h("div", { class: "meta" }, `${c.title} · ${c.source} · distance ${c.distance}`),
    h("div", { class: "text" }, c.content));
}

function card(t) {
  const status = h("span", { class: "status" });
  const note = h("textarea", { placeholder: "What do you notice? (free-text — no categories yet)" });
  note.value = t.note || "";
  autosave(note, t.id, status);

  const top = t.chunks[0]?.distance;
  const gated = top == null || top > THRESHOLD;
  const summary = `Retrieved chunks (${t.chunks.length})` +
    (top != null ? ` · top distance ${top}` : "") + (gated ? " — gated (refused)" : "");

  const left = h("div", {},
    h("label", {}, "Open-code note"), note, status);
  const right = h("div", {},
    h("label", {}, "Answer"),
    h("div", { class: "answer", html: marked.parse(t.answer || "*(no answer)*") }),
    h("details", {},
      h("summary", { class: gated ? "gated" : "" }, summary),
      ...t.chunks.map(chunkEl)));

  return h("div", { class: "card" },
    h("div", {}, h("strong", { style: "margin-right:1rem" }, "#" + t.id), h("span", { class: "q" }, t.question)),
    h("div", { class: "grid" }, left, right));
}

function download(traces) {
  const body = traces.map(t => JSON.stringify({ id: t.id, question: t.question, note: t.note || "" })).join("\\n") + "\\n";
  const a = h("a", { href: URL.createObjectURL(new Blob([body], { type: "application/x-ndjson" })),
                     download: "open_codes.jsonl" });
  a.click(); URL.revokeObjectURL(a.href);
}

fetch("/api/traces").then(r => r.json()).then(({ source, traces, can_pull }) => {
  const app = document.getElementById("app");
  const coded = traces.filter(t => (t.note || "").trim()).length;
  const tools = h("div", { class: "toolbar" },
    h("button", { onclick: () => download(traces) }, "⬇ Download open_codes.jsonl"));
  if (can_pull) {
    const btn = h("button", {}, "↻ Pull new");
    btn.onclick = () => { btn.disabled = true; btn.textContent = "pulling…";
      fetch("/api/pull", { method: "POST" }).then(() => location.reload()); };
    tools.append(btn);
  }
  tools.append(h("span", { class: "muted" }, `${source} · ${coded} / ${traces.length} coded · autosaves as you type`));
  app.append(h("h1", {}, "Trace viewer — open coding"), tools);
  if (!traces.length)
    app.append(h("p", { class: "muted" }, "No traces yet — run: uv run python -m evals.viewers.trace_viewer pull"));
  traces.forEach(t => app.append(card(t)));
});
</script>
"""
HTML = HTML.replace("%THRESHOLD%", str(RELEVANCE_THRESHOLD))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pull":
        pull()
    else:
        uvicorn.run(app, host="127.0.0.1", port=5003)
