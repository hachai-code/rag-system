"""FastAPI wrapper: POST a question, get a grounded answer. See README "API".

Run: uv run fastapi dev rag/app.py
"""

import json
import os
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langfuse import get_client
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .db import connect
from .query.answer import ANSWER_FORMAT, GEN_PROVIDER, answer, answer_stream
from .query.retrieve import RELEVANCE_THRESHOLD, rerank_search, search, source_passage

# Cap the one caller-controlled cost lever before it reaches Voyage/Claude.
MAX_QUESTION_CHARS = 1000

NO_ANSWER = "I don't have information on that in the innerdance corpus."

# Rate limit per client IP. In-memory storage → per-process; point Limiter at Redis
# (storage_uri=...) to share across workers.
limiter = Limiter(key_func=get_remote_address)
RATE_LIMIT = "10/minute"

app = FastAPI(title="innerdance RAG")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: the frontend runs on a different origin. Set FRONTEND_ORIGIN (comma-separated)
# to the deployed URL(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Langfuse tracing: with the LANGFUSE_* keys set, each /ask is one trace; the
# generation call is auto-captured by the active provider's instrumentor (see README).
# Without keys, get_client() is a disabled no-op, so the app runs unchanged.
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    if GEN_PROVIDER == "anthropic":
        AnthropicInstrumentor().instrument()
    else:
        import langfuse.openai  # noqa: F401  (import patches the openai module)
langfuse = get_client()


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]
    # "prose" or "claims"; only affects the openai-compat path (see answer.py).
    format: Literal["prose", "claims"] = ANSWER_FORMAT


class Source(BaseModel):
    title: str
    source: str
    distance: float | None  # None for keyword-only hits (no vector distance)


class Citation(BaseModel):
    claim: str  # the span of the answer this citation backs
    cited_text: str  # the exact source quote
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
    """Compact retrieval summary for the trace (full chunk text would bloat spans)."""
    return [
        {"id": h["id"], "title": h["title"],
         "distance": round(h["distance"], 4) if h.get("distance") is not None else None}
        for h in hits
    ]


# Sync `def`: the Voyage/Claude/psycopg calls block, so FastAPI runs this in a
# threadpool. `request: Request` is unused by the body but required for slowapi.
@app.post("/ask")
@limiter.limit(RATE_LIMIT)
def ask(request: Request, body: AskRequest) -> AskResponse:
    with langfuse.start_as_current_observation(
        as_type="span", name="rag-ask", input=body.question
    ) as span:
        with connect() as conn:
            gate = search(conn, body.question, k=1)  # cheap coverage check only
            if _no_relevant_hits(gate):
                span.update(metadata={"retrieved": _retrieved_meta(gate)}, output=NO_ANSWER)
                return AskResponse(answer=NO_ANSWER, citations=[], sources=[])
            hits = rerank_search(conn, body.question)  # hybrid + RRF + rerank
        span.update(metadata={"retrieved": _retrieved_meta(hits)})
        text, citations = answer(body.question, hits, fmt=body.format)
        span.update(output=text)
        return AskResponse(
            answer=text,
            citations=[Citation(**c) for c in citations],
            sources=[
                Source(title=h["title"], source=h["source"], distance=h.get("distance"))
                for h in hits
            ],
        )


# Server-Sent Events: `text` events stream in, then one `citation` event per source.
@app.post("/ask/stream")
@limiter.limit(RATE_LIMIT)
def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    # The span lives inside the generator so the streamed generation nests under it.
    def events():
        with langfuse.start_as_current_observation(
            as_type="span", name="rag-ask-stream", input=body.question
        ) as span:
            with connect() as conn:
                gate = search(conn, body.question, k=1)  # cheap coverage check only
                if _no_relevant_hits(gate):
                    span.update(metadata={"retrieved": _retrieved_meta(gate)}, output=NO_ANSWER)
                    yield f"data: {json.dumps({'type': 'text', 'text': NO_ANSWER})}\n\n"
                    return
                hits = rerank_search(conn, body.question)  # hybrid + RRF + rerank
            span.update(metadata={"retrieved": _retrieved_meta(hits)})
            answer_text = []
            for event in answer_stream(body.question, hits, fmt=body.format):
                if event["type"] == "text":
                    answer_text.append(event["text"])
                yield f"data: {json.dumps(event)}\n\n"
            span.update(output="".join(answer_text))

    return StreamingResponse(events(), media_type="text/event-stream")


# The chunk a citation points at, reconstructed in its place for the frontend.
@app.get("/source/{chunk_id}")
@limiter.limit(RATE_LIMIT)
def source(request: Request, chunk_id: int) -> SourcePassage:
    with connect() as conn:
        return SourcePassage(**source_passage(conn, chunk_id))
