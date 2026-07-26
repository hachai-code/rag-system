import json
from pathlib import Path

from fasthtml.common import *

# Axial failure categories A–F. One home: evals/taxonomy.json (also drives the judge
# rubrics). Each is one binary judge; severity drives prioritization (A and D are the
# safety-critical pair — fail them hard even when the rest of the answer is great).
TAXONOMY = json.loads((Path(__file__).resolve().parents[2] / "taxonomy.json").read_text())
CODES = list(TAXONOMY)
DIFFICULTIES = ["easy", "medium", "hard"]
SPLITS = ["dev", "test"]

app, rt = fast_app()

# ponytail: in-memory store for a local single-user tool. A browser page can't write
# back to the uploaded file's location, so: upload to load, Download to save.
ENTRIES = []
FILENAME = "eval.jsonl"
LOADED = False

DROP_JS = """
const dz = document.getElementById('dz'), f = document.getElementById('file');
dz.onclick = () => f.click();
dz.ondragover = e => { e.preventDefault(); dz.style.background = '#eef'; };
dz.ondragleave = () => dz.style.background = '';
dz.ondrop = e => { e.preventDefault(); f.files = e.dataTransfer.files; f.form.submit(); };
"""


def dropzone(error=None):
    return Titled(
        "RAG Eval — drop a JSONL to load",
        P(error, style="color:red") if error else "",
        Form(
            Input(
                type="file",
                name="file",
                accept=".jsonl",
                id="file",
                onchange="this.form.submit()",
                style="display:none",
            ),
            Div(
                "Drop a .jsonl here, or click to browse",
                id="dz",
                style="border:2px dashed #888; padding:3rem; text-align:center; cursor:pointer",
            ),
            method="post",
            action=upload,
        ),
        P(A("…or start with a blank file", href=blank)),
        Script(DROP_JS),
    )


def taxonomy_legend():
    blocks = []
    for code, t in TAXONOMY.items():
        critical = t["severity"] == "safety-critical"
        blocks.append(
            Article(
                H4(f"{code} — {t['name']}"),
                P(
                    B(t["severity"], style="color:#c00") if critical else Small(t["severity"]),
                    Small(f" · {t['layer']} layer · folds in {t['folds']}"),
                    style="margin-top:-0.5rem",
                ),
                P(t["desc"]),
            )
        )
    return Details(
        Summary("Failure taxonomy — what the axial codes (A–F) mean"),
        P(
            "These are the generation/answer-quality failure categories from a three-pass "
            "qualitative coding of human reviews (15 open codes folded into 6 axial categories). "
            "Each becomes one binary judge. Pick every category an entry is meant to probe."
        ),
        *blocks,
        P(
            Small(
                "A and D are the safety-critical pair — a leak (A) or an unverified "
                "medical-adjacent claim (D) fails the answer outright. The set is C-heavy and "
                "starved on A, B, E, so per-category judge accuracy isn't yet trustworthy: "
                "grow A (extraction red-team) and D (medical) before relying on it."
            ),
            style="color:#888",
        ),
        open=True,
    )


@rt
def index():
    if not LOADED:
        return dropzone()
    form = Form(
        Textarea(name="question", placeholder="Question", required=True, rows=2),
        Textarea(name="ideal_answer", placeholder="Ideal answer (optional)", rows=3),
        Fieldset(
            *[
                Label(
                    Input(type="checkbox", name="axial_codes", value=c, title=TAXONOMY[c]["name"]),
                    c,
                )
                for c in CODES
            ],
            style="display:flex; gap:1rem",
        ),
        Label("Difficulty", Select(*map(Option, DIFFICULTIES), name="difficulty")),
        Label("Split", Select(*map(Option, SPLITS), name="split")),
        Button("Add entry"),
        method="post",
        action=add,
    )
    rows = [
        Tr(
            Td(e["id"]),
            Td(e["question"]),
            Td(", ".join(e["axial_codes"])),
            Td(e["difficulty"]),
            Td(e["split"]),
        )
        for e in ENTRIES
    ]
    table = Table(
        Thead(Tr(*map(Th, ["id", "question", "codes", "difficulty", "split"]))),
        Tbody(*rows),
    )
    return Titled(
        f"{FILENAME} — {len(ENTRIES)} entries",
        P(
            A("⬇ Download .jsonl", href=download, role="button"),
            " ",
            A("load a different file", href=reset),
        ),
        P(Small("Held in memory — click Download to save your changes."), style="color:#888"),
        taxonomy_legend(),
        form,
        H2("Entries"),
        table,
    )


@rt
async def upload(file: UploadFile):
    global ENTRIES, FILENAME, LOADED
    text = (await file.read()).decode()
    try:
        ENTRIES = [json.loads(l) for l in text.splitlines() if l.strip()]
    except Exception as e:
        return dropzone(error=f"{file.filename}: not valid JSONL ({e})")
    FILENAME = file.filename or "eval.jsonl"
    LOADED = True
    return Redirect(index)


@rt
def blank():
    global ENTRIES, FILENAME, LOADED
    ENTRIES, FILENAME, LOADED = [], "eval.jsonl", True
    return Redirect(index)


@rt
def reset():
    global LOADED
    LOADED = False
    return Redirect(index)


@rt
def add(
    question: str,
    difficulty: str,
    split: str,
    ideal_answer: str = "",
    axial_codes: list[str] = None,
):
    ENTRIES.append(
        {
            "id": max((e["id"] for e in ENTRIES), default=0) + 1,
            "question": question,
            "ideal_answer": ideal_answer,
            "axial_codes": axial_codes or [],
            "difficulty": difficulty,
            "split": split,
        }
    )
    return Redirect(index)


@rt
def download():
    body = "".join(json.dumps(e) + "\n" for e in ENTRIES)
    return Response(
        body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{FILENAME}"'},
    )


serve()
