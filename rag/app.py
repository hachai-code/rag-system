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
from .query.answer import ANSWER_FORMAT, GEN_MODELS, GEN_PROVIDER, answer, answer_stream
from .query.retrieve import (
    HYPE,
    METHOD,
    PARENT_DOCUMENT,
    QUERY_ENHANCEMENT,
    RELEVANCE_THRESHOLD,
    RERANK_DEPTH,
    TOP_K,
    retrieve,
    search,
    source_passage,
)
from .query.deepagent import resume_deepagent, run_deepagent, stream_deepagent
from .query.web_search_graph_agent import stream_agent

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
    # Generation model picker; resolved to an OpenRouter id via GEN_MODELS.
    model: Literal["pro", "flash"] = "pro"
    # Retriever funnel; default from config.toml ([retrieval] method), production is rerank.
    method: Literal["vector", "hybrid", "rerank"] = METHOD
    # Runtime query rewriting: HyDE hypothetical or multi-query paraphrase fusion (off by default).
    query_enhancement: Literal["hyde", "multi_query"] | None = QUERY_ENHANCEMENT
    # Widen each hit to its neighbouring chunks before answering (parent-document retrieval).
    parent_document: bool = PARENT_DOCUMENT
    # Match the query against index-time hypothetical questions instead of raw chunks (HyPE).
    hype: bool = HYPE
    # Chunks handed to the generator (the citable pool). Defaults to config top_k;
    # can't exceed RERANK_DEPTH, since rerank only has that many candidates to keep.
    top_k: Annotated[int, Field(ge=1, le=RERANK_DEPTH)] = TOP_K


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
            hits = retrieve(conn, body.question, k=body.top_k, method=body.method,
                            query_enhancement=body.query_enhancement,
                            parent_document=body.parent_document, hype=body.hype)
        span.update(metadata={"retrieved": _retrieved_meta(hits)})
        text, citations = answer(body.question, hits, model=GEN_MODELS[body.model], fmt=body.format)
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
                hits = retrieve(conn, body.question, k=body.top_k, method=body.method,
                                query_enhancement=body.query_enhancement,
                                parent_document=body.parent_document, hype=body.hype)
            span.update(metadata={"retrieved": _retrieved_meta(hits)})
            answer_text = []
            for event in answer_stream(body.question, hits, model=GEN_MODELS[body.model], fmt=body.format):
                if event["type"] == "text":
                    answer_text.append(event["text"])
                yield f"data: {json.dumps(event)}\n\n"
            span.update(output="".join(answer_text))

    return StreamingResponse(events(), media_type="text/event-stream")


class AgentRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]


# SSE: step_started / answer_token / tool_call / tool_result / critique events
# stream in as the agent works, terminated by one done (or error) event.
@app.post("/agent/stream")
@limiter.limit(RATE_LIMIT)
def agent_stream(request: Request, body: AgentRequest) -> StreamingResponse:
    def events():
        for event in stream_agent(body.question):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


class DeepAgentRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]
    # The research thread this run belongs to; keys the durable state (Phase 2+).
    thread_id: Annotated[str, Field(min_length=1)]


class DeepAgentResponse(BaseModel):
    answer: str
    thread_id: str


# Returned when the HITL gate (config: enable_hitl) pauses the run before external
# research. A later POST /ask/agent/resume on the same thread_id continues it.
class AwaitingApproval(BaseModel):
    status: Literal["awaiting_approval"] = "awaiting_approval"
    thread_id: str
    pending: list[dict[str, str]]


class ResumeRequest(BaseModel):
    thread_id: Annotated[str, Field(min_length=1)]
    decision: Literal["approve", "reject"]


# The Deep Agent path: answers from the corpus via its own tool-driven control
# flow. The run is traced (span lives in deepagent.run_deepagent).
@app.post("/ask/agent")
@limiter.limit(RATE_LIMIT)
def ask_deepagent(request: Request, body: DeepAgentRequest) -> DeepAgentResponse | AwaitingApproval:
    result = run_deepagent(body.question, body.thread_id)
    if result["status"] == "awaiting_approval":
        return AwaitingApproval(thread_id=result["thread_id"], pending=result["pending"])
    return DeepAgentResponse(answer=result["answer"], thread_id=result["thread_id"])


# Resume a paused thread: approve or reject the external-research gate. Works from a
# separate request even after a restart (state is durable in Postgres).
@app.post("/ask/agent/resume")
@limiter.limit(RATE_LIMIT)
def resume_deepagent_endpoint(
    request: Request, body: ResumeRequest
) -> DeepAgentResponse | AwaitingApproval:
    result = resume_deepagent(body.thread_id, body.decision)
    if result["status"] == "awaiting_approval":
        return AwaitingApproval(thread_id=result["thread_id"], pending=result["pending"])
    return DeepAgentResponse(answer=result["answer"], thread_id=result["thread_id"])


# SSE: `status` events stream in as the deep agent retrieves, plans, and delegates
# web research (subagent steps included), terminated by one `answer` (or `error`).
@app.post("/ask/agent/stream")
@limiter.limit(RATE_LIMIT)
def ask_deepagent_stream(request: Request, body: DeepAgentRequest) -> StreamingResponse:
    def events():
        for event in stream_deepagent(body.question, body.thread_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# The chunk a citation points at, reconstructed in its place for the frontend.
@app.get("/source/{chunk_id}")
@limiter.limit(RATE_LIMIT)
def source(request: Request, chunk_id: int) -> SourcePassage:
    with connect() as conn:
        return SourcePassage(**source_passage(conn, chunk_id))
