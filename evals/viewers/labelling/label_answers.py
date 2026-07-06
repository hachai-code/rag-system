import json
from datetime import datetime
from pathlib import Path

from fasthtml.common import *

app, rt = fast_app(hdrs=(MarkdownJS(),))  # renders elements with class "marked"

# Single-user local tool. Drop a .jsonl to load, edit labels inline, Download to
# save a copy. State is also written through to WORK on every save so labels
# survive a server restart (the browser can't give the uploaded file's path).
# Records: {id, question, rag_answer, label}.
WORK = Path(__file__).parent / ".labels_state.json"
ENTRIES = []
FILENAME = "labels.jsonl"
LOADED = False

if WORK.exists():
    _state = json.loads(WORK.read_text())
    ENTRIES, FILENAME, LOADED = _state["entries"], _state["filename"], True


def persist():
    WORK.write_text(json.dumps({"filename": FILENAME, "entries": ENTRIES}))

DROP_JS = """
const dz = document.getElementById('dz'), f = document.getElementById('file');
dz.onclick = () => f.click();
dz.ondragover = e => { e.preventDefault(); dz.style.background = '#eef'; };
dz.ondragleave = () => dz.style.background = '';
dz.ondrop = e => { e.preventDefault(); f.files = e.dataTransfer.files; f.form.submit(); };
"""


def dropzone(error=None):
    return Titled(
        "Answer labelling — drop a JSONL to load",
        P("Expects records with question, rag_answer, and label fields."),
        P(error, style="color:red") if error else "",
        Form(
            Input(type="file", name="file", accept=".jsonl", id="file",
                  onchange="this.form.submit()", style="display:none"),
            Div("Drop a .jsonl here, or click to browse", id="dz",
                style="border:2px dashed #888; padding:3rem; text-align:center; cursor:pointer"),
            method="post", action=upload,
        ),
        Script(DROP_JS),
    )


def find(id):
    return next((e for e in ENTRIES if e["id"] == id), None)


def field(e, name, **kw):
    # Autosave textarea: posts ~0.5s after typing pauses (and on blur), swapping
    # only the status span (never itself), so the cursor isn't disturbed.
    return Textarea(e[name], name=name,
                    hx_post=f"/save/{e['id']}", hx_trigger="input changed delay:500ms, blur",
                    hx_target=f"#status-{e['id']}", hx_swap="innerHTML", **kw)


ANS_BOX = ("max-height:32rem; overflow:auto; line-height:1.6; color:#000; background:#fff; "
           "padding:1rem 1.25rem; border-radius:6px; border:1px solid #ccc")


def answer_view(e):
    i = e["id"]
    return Div(
        Div(e["rag_answer"] or "(no answer)", cls="marked", style=ANS_BOX),
        A("edit", hx_get=f"/edit_ans/{i}", hx_target=f"#ans-{i}", hx_swap="outerHTML",
          style="cursor:pointer"),
        id=f"ans-{i}",
    )


def answer_edit(e):
    i = e["id"]
    return Div(
        field(e, "rag_answer", style="width:100%; height:24rem"),
        A("done", hx_get=f"/view_ans/{i}", hx_target=f"#ans-{i}", hx_swap="outerHTML",
          style="cursor:pointer"),
        id=f"ans-{i}",
    )


def card(e):
    i = e["id"]
    left = Div(
        Div(H4(f"#{i}", style="display:inline; margin-right:1rem"),
            Button("delete", hx_post=f"/delete/{i}", hx_target=f"#row-{i}", hx_swap="outerHTML",
                   hx_confirm="Delete this entry?", cls="secondary outline",
                   style="display:inline; width:auto; padding:0.1rem 0.6rem")),
        Label("Question", field(e, "question", rows=2, style="width:100%")),
        Label("Label", field(e, "label", style="width:100%; height:20rem",
                             placeholder="Your label / judgment…")),
        Small(id=f"status-{i}", style="color:green"),
    )
    right = Div(Label("Answer"), answer_view(e))
    return Article(
        Div(left, right, style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem"),
        id=f"row-{i}",
    )


@rt
def index():
    if not LOADED:
        return dropzone()
    labelled = sum(1 for e in ENTRIES if e["label"])
    return Titled(
        f"{FILENAME} — labelling",
        P(A("⬇ Download .jsonl", href=download, role="button"), " ",
          Form(Button("+ Add entry"), method="post", action=add, style="display:inline"), " ",
          A("load a different file", href=reset)),
        P(B(f"{labelled} / {len(ENTRIES)} labelled"),
          Small("  ·  autosaves as you type; survives restarts; Download for a file copy", style="color:#888")),
        *[card(e) for e in ENTRIES],
    )


@rt
async def upload(file: UploadFile):
    global ENTRIES, FILENAME, LOADED
    text = (await file.read()).decode()
    try:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    except Exception as e:
        return dropzone(error=f"{file.filename}: not valid JSONL ({e})")
    ENTRIES = [{"id": r.get("id"), "question": r.get("question", ""),
                "rag_answer": r.get("rag_answer", ""), "label": r.get("label", "")}
               for r in rows]
    FILENAME = file.filename or "labels.jsonl"
    LOADED = True
    persist()
    return Redirect(index)


@rt("/save/{id}")
def save(id: int, question: str = None, rag_answer: str = None, label: str = None):
    e = next((x for x in ENTRIES if x["id"] == id), None)
    if e is None:
        return ""
    for k, v in (("question", question), ("rag_answer", rag_answer), ("label", label)):
        if v is not None:  # autosave posts only the one field that changed
            e[k] = v
    persist()
    return f"saved ✓ {datetime.now():%H:%M:%S}"


@rt
def add():
    ENTRIES.append({"id": max((e["id"] for e in ENTRIES), default=0) + 1,
                    "question": "", "rag_answer": "", "label": ""})
    persist()
    return Redirect(index)


@rt("/edit_ans/{id}")
def edit_ans(id: int):
    e = find(id)
    return answer_edit(e) if e else ""


@rt("/view_ans/{id}")
def view_ans(id: int):
    e = find(id)
    return answer_view(e) if e else ""


@rt("/delete/{id}")
def del_entry(id: int):  # not named "delete": FastHTML would bind that to HTTP DELETE
    global ENTRIES
    ENTRIES = [e for e in ENTRIES if e["id"] != id]
    persist()
    return ""


@rt
def reset():
    global ENTRIES, FILENAME, LOADED
    ENTRIES, FILENAME, LOADED = [], "labels.jsonl", False
    WORK.unlink(missing_ok=True)
    return Redirect(index)


@rt
def download():
    body = "".join(json.dumps(e) + "\n" for e in ENTRIES)
    return Response(body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="{FILENAME}"'})


serve(port=5002)
