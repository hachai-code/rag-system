"""FastAPI wrapper around the RAG pipeline: POST a question, get a grounded answer.

Run: uv run fastapi dev app.py
Then: curl -X POST localhost:8000/ask -H 'content-type: application/json' \
        -d '{"question": "..."}'
"""

import json
import os

import psycopg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel

from rag import DB_URL, answer, answer_stream, search, source_passage

app = FastAPI(title="innerdance RAG")

# The frontend runs on a different origin, so the browser needs CORS. Defaults to
# the local dev server; set FRONTEND_ORIGIN (comma-separated for more than one) to
# the deployed frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    source: str
    distance: float


class Citation(BaseModel):
    claim: str  # the span of the answer this citation backs
    cited_text: str  # the exact source quote, extracted by the API
    chunk_id: int
    title: str
    source: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[Source]


class SourcePassage(BaseModel):
    title: str
    section: str
    chunk_index: int
    n_chunks: int
    before: str  # context preceding the cited chunk
    chunk: str  # the retrieved chunk, to highlight
    after: str  # context following the cited chunk


# Sync `def` (not async): the Voyage/Claude/psycopg calls block, so FastAPI runs
# this in a threadpool instead of stalling the event loop.
@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = search(conn, request.question)
    text, citations = answer(request.question, hits)
    return AskResponse(
        answer=text,
        citations=[Citation(**c) for c in citations],
        sources=[
            Source(title=h["title"], source=h["source"], distance=h["distance"])
            for h in hits
        ],
    )


# Server-Sent Events: the answer streams in as `text` events, then one `citation`
# event per source once the message completes. The frontend reads these live.
@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = search(conn, request.question)

    def events():
        for event in answer_stream(request.question, hits):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# Click-through to source: the chunk a citation points at, reconstructed in its
# place in the document so the frontend can highlight it in context.
@app.get("/source/{chunk_id}")
def source(chunk_id: int) -> SourcePassage:
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        return SourcePassage(**source_passage(conn, chunk_id))
