"""Open-coding viewer over REAL production traces from Langfuse — not the eval set.

Where trace_viewer.py replays eval questions through the pipeline, this reads what
actually happened: the live /ask endpoint logs every query to Langfuse (app.py) with
the question (input), the answer (output), and the retrieved chunk ids (metadata).
We pull those traces and rehydrate the chunk *content* from Postgres by id — the trace
stores only id/title/distance to keep spans small — producing the same
{id, question, chunks, answer, note} shape trace_viewer's UI already renders.

Reads the rag-ask/rag-ask-stream observations (via evals.flywheel.sample_traces.spans) rather
than trace-level fields: since FastAPI auto-instrumentation the trace root is the
HTTP span, so question/answer/metadata live on the rag-ask span. No LLM calls here —
pulling is just a Langfuse read plus a chunk-content lookup, so it's cheap and safe
to re-run.

    uv run python -m evals.viewers.langfuse_viewer pull   # cache real traces -> langfuse_traces.jsonl
    uv run python -m evals.viewers.langfuse_viewer        # serve the viewer (default), port 5004
"""

import json
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from langfuse import get_client

from evals.flywheel.sample_traces import SPAN_NAMES, spans
from evals.viewers.trace_viewer import HTML, load  # same open-coding UI, different data source
from rag.db import connect

load_dotenv()
HERE = Path(__file__).parent
TRACES = HERE / "langfuse_traces.jsonl"


def chunk_rows(conn, ids: list[int]) -> dict:
    """title/source/content for the given chunk ids (content lives in chunks, the rest
    in documents — same join rag.search uses)."""
    if not ids:
        return {}
    rows = conn.execute(
        """SELECT c.id, d.title, d.source, c.content
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.id = ANY(%s)""",
        (ids,),
    ).fetchall()
    return {r["id"]: r for r in rows}


def pull() -> None:
    """Cache real Langfuse traces, rehydrating chunk content. Resumable by trace id."""
    done = {t["id"] for t in load(TRACES)}
    lf = get_client()
    traces = [o for name in SPAN_NAMES for o in spans(lf, name) if o.trace_id not in done]
    if not traces:
        print(f"{TRACES}: already complete ({len(done)} traces)")
        return
    with connect() as conn, TRACES.open("a") as out:
        for t in traces:
            retrieved = (t.metadata or {}).get("retrieved", [])
            content = chunk_rows(conn, [r["id"] for r in retrieved])
            chunks = [
                {
                    "title": r["title"],
                    "source": content.get(r["id"], {}).get("source", ""),
                    "distance": r["distance"],
                    "content": content.get(r["id"], {}).get(
                        "content", "(chunk no longer in corpus)"
                    ),
                }
                for r in retrieved
            ]
            out.write(
                json.dumps(
                    {
                        "id": t.trace_id,
                        "question": str(t.input) if t.input else "",
                        "chunks": chunks,
                        "answer": str(t.output) if t.output else "",
                        "note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            print(f"  pulled {t.trace_id[:8]}…  {len(chunks)} chunks  {str(t.input)[:50]}")
    print(f"{TRACES}: {len(load(TRACES))} traces")


app = FastAPI()


@app.get("/api/traces")
def api_traces() -> dict:
    return {"source": "Langfuse (production)", "traces": load(TRACES), "can_pull": True}


@app.post("/api/pull")
def api_pull() -> dict:
    """Fetch any traces logged since the last pull. Resumable, so this only adds new
    ones — cheap (a Langfuse read + chunk lookup), which is why it's a button."""
    before = len(load(TRACES))
    pull()
    return {"added": len(load(TRACES)) - before}


@app.post("/api/note/{tid}")
async def api_note(tid: str, req: Request) -> dict:  # Langfuse trace ids are strings, not ints
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pull":
        pull()
    else:
        uvicorn.run(app, host="127.0.0.1", port=5004)
