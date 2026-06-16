"""FastAPI wrapper around the RAG pipeline: POST a question, get a grounded answer.

Run: uv run fastapi dev app.py
Then: curl -X POST localhost:8000/ask -H 'content-type: application/json' \
        -d '{"question": "..."}'
"""

import json
import os
from typing import Annotated

import psycopg
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langfuse import get_client
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag import (
    DB_URL,
    RELEVANCE_THRESHOLD,
    answer,
    answer_stream,
    search,
    source_passage,
)

# Reject over-long questions before they reach Voyage/Claude: a long prompt costs
# more to embed and generate over, and is the one input a caller controls, so it's
# the cost lever worth bounding. 1000 chars is generous for a real question.
MAX_QUESTION_CHARS = 1000

# What we return when retrieval finds nothing relevant (see RELEVANCE_THRESHOLD):
# refuse instead of letting Claude answer from nothing.
NO_ANSWER = "I don't have information on that in the innerdance corpus."

# Rate limit per client IP. Storage is in-memory, so it's per-process — fine for a
# single worker; point Limiter at Redis (storage_uri=...) if you run several.
limiter = Limiter(key_func=get_remote_address)
RATE_LIMIT = "10/minute"

app = FastAPI(title="innerdance RAG")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# The frontend runs on a different origin, so the browser needs CORS. Defaults to
# the local dev server; set FRONTEND_ORIGIN (comma-separated for more than one) to
# the deployed frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tracing: with the LANGFUSE_* keys set, each /ask is one Langfuse trace — the
# retrieved chunks as metadata, plus the Claude call, which the Anthropic
# instrumentor captures automatically (model, prompt, output, token usage). Without
# keys, get_client() returns a disabled client and the spans below become no-ops, so
# the app (and the tests that import it) run unchanged. The server is long-lived, so
# we don't flush per request — the SDK batches in the background and flushes at exit.
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    AnthropicInstrumentor().instrument()
langfuse = get_client()


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]


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


def _no_relevant_hits(hits: list[dict]) -> bool:
    """True when retrieval found nothing close enough to answer from."""
    return not hits or hits[0]["distance"] > RELEVANCE_THRESHOLD


def _retrieved_meta(hits: list[dict]) -> list[dict]:
    """Compact summary of what retrieval returned, for the trace — the full chunk
    text would bloat every span, so we keep just id, title, and distance."""
    return [
        {"id": h["id"], "title": h["title"], "distance": round(h["distance"], 4)}
        for h in hits
    ]


# Sync `def` (not async): the Voyage/Claude/psycopg calls block, so FastAPI runs
# this in a threadpool instead of stalling the event loop. `request: Request` is
# unused by the body but required for slowapi to read the client IP.
@app.post("/ask")
@limiter.limit(RATE_LIMIT)
def ask(request: Request, body: AskRequest) -> AskResponse:
    with langfuse.start_as_current_observation(
        as_type="span", name="rag-ask", input=body.question
    ) as span:
        with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
            register_vector(conn)
            hits = search(conn, body.question)
        span.update(metadata={"retrieved": _retrieved_meta(hits)})
        if _no_relevant_hits(hits):
            span.update(output=NO_ANSWER)
            return AskResponse(answer=NO_ANSWER, citations=[], sources=[])
        text, citations = answer(body.question, hits)
        span.update(output=text)
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
@limiter.limit(RATE_LIMIT)
def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    # The span lives inside the generator: the Claude call happens lazily as the
    # response streams, so the trace has to stay open until the stream is done for
    # the generation to nest under it. Retrieval moves in here too, so it's traced.
    def events():
        with langfuse.start_as_current_observation(
            as_type="span", name="rag-ask-stream", input=body.question
        ) as span:
            with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
                register_vector(conn)
                hits = search(conn, body.question)
            span.update(metadata={"retrieved": _retrieved_meta(hits)})
            if _no_relevant_hits(hits):
                span.update(output=NO_ANSWER)
                yield f"data: {json.dumps({'type': 'text', 'text': NO_ANSWER})}\n\n"
                return
            answer_text = []
            for event in answer_stream(body.question, hits):
                if event["type"] == "text":
                    answer_text.append(event["text"])
                yield f"data: {json.dumps(event)}\n\n"
            span.update(output="".join(answer_text))

    return StreamingResponse(events(), media_type="text/event-stream")


# Click-through to source: the chunk a citation points at, reconstructed in its
# place in the document so the frontend can highlight it in context.
@app.get("/source/{chunk_id}")
@limiter.limit(RATE_LIMIT)
def source(request: Request, chunk_id: int) -> SourcePassage:
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        return SourcePassage(**source_passage(conn, chunk_id))
