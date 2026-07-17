"""FastAPI wrapper: POST a question, get a grounded answer. See README "API".

Run: uv run fastapi dev rag/app.py
"""

import os
import threading
import time
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langfuse import Langfuse, get_client
from langfuse.span_filter import is_default_export_span
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field, RootModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from evals.api import router as evals_router

from .db import Hit, connect
from .guardrails import BLOCKED, check_input, check_output
from .query.agent import resume_deepagent, run_deepagent, stream_agent, stream_deepagent
from .query.answer import ANSWER_FORMAT, GEN_MODELS, GEN_PROVIDER, answer_stream
from .query.gate import NO_ANSWER, ask_gate, gate_retrieve
from .query.memory import get_memory, list_memories
from .query.retrieve import (
    HYPE,
    METHOD,
    PARENT_DOCUMENT,
    QUERY_ENHANCEMENT,
    RERANK_DEPTH,
    TOP_K,
)
from .query.sources import source_passage

# Cap the one caller-controlled cost lever before it reaches Voyage/Claude.
MAX_QUESTION_CHARS = 1000

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

app.include_router(evals_router)

# Langfuse tracing: with the LANGFUSE_* keys set, each /ask is one trace; the
# generation call is auto-captured by the active provider's instrumentor.
# Without keys, get_client() is a disabled no-op, so the app runs unchanged.
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    if GEN_PROVIDER == "anthropic":
        AnthropicInstrumentor().instrument()
    else:
        import langfuse.openai  # noqa: F401  (import patches the openai module)
    FastAPIInstrumentor.instrument_app(
        app, excluded_urls="docs,openapi.json,redoc", exclude_spans=["receive", "send"]
    )
    # Langfuse only exports LLM-ish spans by default; allowlist the FastAPI HTTP
    # spans too. Langfuse(...) registers the singleton get_client() returns elsewhere.
    langfuse = Langfuse(
        should_export_span=lambda s: (
            is_default_export_span(s)
            or (
                s.instrumentation_scope is not None
                and s.instrumentation_scope.name == "opentelemetry.instrumentation.fastapi"
            )
        )
    )
else:
    langfuse = get_client()


def sse(events: Iterable[dict]) -> EventSourceResponse:
    """Frame an event stream as SSE — the one framing every streaming route uses.

    JSONServerSentEvent, because sse-starlette reads a bare dict as ServerSentEvent
    kwargs rather than as the payload. sep="\\n" on both the events and the response
    keeps every frame LF-terminated as before (the library defaults to CRLF, and an
    event's own separator wins over the response's). Events are sync generators, which
    sse-starlette runs in a threadpool, same as StreamingResponse did.
    """
    return EventSourceResponse((JSONServerSentEvent(e, sep="\n") for e in events), sep="\n")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    format: Literal["prose", "claims"] = Field(
        ANSWER_FORMAT, description="Answer format; only affects the openai-compat path."
    )
    model: Literal["pro", "flash"] = Field(
        "pro", description="Generation model picker, resolved to an OpenRouter id via GEN_MODELS."
    )
    method: Literal["vector", "hybrid", "rerank"] = Field(
        METHOD, description="Retriever funnel; production default is rerank."
    )
    query_enhancement: Literal["hyde", "multi_query"] | None = Field(
        QUERY_ENHANCEMENT,
        description="Query rewriting: HyDE hypothetical or multi-query paraphrase fusion.",
    )
    parent_document: bool = Field(
        PARENT_DOCUMENT,
        description="Widen each hit to its neighbouring chunks before answering.",
    )
    hype: bool = Field(
        HYPE,
        description="Match against index-time hypothetical questions instead of raw chunks.",
    )
    top_k: int = Field(
        TOP_K,
        ge=1,
        le=RERANK_DEPTH,
        description="Chunks handed to the generator (the citable pool); "
        "can't exceed RERANK_DEPTH, since rerank only has that many candidates to keep.",
    )


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


class TextEvent(BaseModel):
    """A slice of the answer, as it generates."""

    type: Literal["text"] = "text"
    text: str


class CitationEvent(Citation):
    """One source backing the answer, sent once the text is complete."""

    type: Literal["citation"] = "citation"


# The wire contract of /ask/stream, declared so it reaches the OpenAPI schema (and
# through it, the frontend's generated types). answer_stream yields these as plain
# dicts; the model documents the shape, it doesn't validate the stream.
class StreamEvent(RootModel):
    """One SSE frame from POST /ask/stream: `text` events, then one `citation` per source."""

    root: Annotated[TextEvent | CitationEvent, Field(discriminator="type")]


def _retrieved_meta(hits: list[Hit]) -> list[dict]:
    """Compact retrieval summary for the trace (full chunk text would bloat spans)."""
    return [
        {
            "id": h["id"],
            "title": h["title"],
            "distance": round(h["distance"], 4) if h.get("distance") is not None else None,
        }
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
        if check_input(body.question):
            span.update(output=BLOCKED)
            return AskResponse(answer=BLOCKED, citations=[], sources=[])
        with connect() as conn:
            result = ask_gate(
                conn,
                body.question,
                k=body.top_k,
                method=body.method,
                query_enhancement=body.query_enhancement,
                parent_document=body.parent_document,
                hype=body.hype,
                model=GEN_MODELS[body.model],
                fmt=body.format,
            )
        span.update(metadata={"retrieved": _retrieved_meta(result.hits)})
        if result.gated:
            span.update(output=NO_ANSWER)
            return AskResponse(answer=NO_ANSWER, citations=[], sources=[])
        if check_output(body.question, result.answer):
            span.update(output=BLOCKED)
            return AskResponse(answer=BLOCKED, citations=[], sources=[])
        span.update(output=result.answer)
        return AskResponse(
            answer=result.answer,
            citations=[Citation(**c) for c in result.citations],
            sources=[
                Source(title=h["title"], source=h["source"], distance=h.get("distance"))
                for h in result.hits
            ],
        )


# Server-Sent Events: `text` events stream in, then one `citation` event per source.
@app.post("/ask/stream", responses={200: {"model": StreamEvent, "description": "SSE stream"}})
@limiter.limit(RATE_LIMIT)
def ask_stream(request: Request, body: AskRequest) -> EventSourceResponse:
    # The span lives inside the generator so the streamed generation nests under it.
    def events():
        with langfuse.start_as_current_observation(
            as_type="span", name="rag-ask-stream", input=body.question
        ) as span:
            # Input rail only: tokens stream straight to the client, so a post-hoc
            # output check couldn't unsend them.
            if check_input(body.question):
                span.update(output=BLOCKED)
                yield {"type": "text", "text": BLOCKED}
                return
            with connect() as conn:
                gated, hits = gate_retrieve(
                    conn,
                    body.question,
                    k=body.top_k,
                    method=body.method,
                    query_enhancement=body.query_enhancement,
                    parent_document=body.parent_document,
                    hype=body.hype,
                )
            span.update(metadata={"retrieved": _retrieved_meta(hits)})
            if gated:
                span.update(output=NO_ANSWER)
                yield {"type": "text", "text": NO_ANSWER}
                return
            answer_text = []
            for event in answer_stream(
                body.question, hits, model=GEN_MODELS[body.model], fmt=body.format
            ):
                if event["type"] == "text":
                    answer_text.append(event["text"])
                yield event
            span.update(output="".join(answer_text))

    return sse(events())


class AgentRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]


# SSE: tool_call / tool_result events stream in as the web agent works, terminated by
# one done (or error) event.
@app.post("/agent/stream")
@limiter.limit(RATE_LIMIT)
def agent_stream(request: Request, body: AgentRequest) -> EventSourceResponse:
    def events():
        if check_input(body.question):
            yield {"type": "error", "message": BLOCKED}
            return
        yield from stream_agent(body.question)

    return sse(events())


class DeepAgentRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]
    # The research thread this run belongs to; keys the durable state (Phase 2+).
    thread_id: Annotated[str, Field(min_length=1)]
    # Max web calls for this run; 0 = unlimited. None falls back to config.
    research_budget: Annotated[int | None, Field(ge=0)] = None


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


class CorpusSource(BaseModel):
    """A corpus passage the deep agent's answer cites. `n` is its stable [n] marker in
    the answer text; chunk_id opens it via GET /source/{chunk_id}."""

    n: int
    chunk_id: int
    title: str
    source: str


class StatusEvent(BaseModel):
    """A tool call the agent started. scope "research" = inside the web-research subagent."""

    type: Literal["status"] = "status"
    scope: Literal["main", "research"]
    call_id: str
    tool: str
    label: str


class ResultEvent(BaseModel):
    """What a tool returned, correlated to its StatusEvent by call_id."""

    type: Literal["result"] = "result"
    call_id: str
    preview: str


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    sources: list[CorpusSource]


class AnswerEvent(BaseModel):
    type: Literal["answer"] = "answer"
    text: str
    thread_id: str


class AwaitingApprovalEvent(BaseModel):
    """The HITL gate paused the run; POST /ask/agent/resume continues it."""

    type: Literal["awaiting_approval"] = "awaiting_approval"
    thread_id: str
    pending: list[dict[str, str]]


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# As with StreamEvent: declared for the OpenAPI schema (and the generated frontend
# types). stream_deepagent yields these as plain dicts — documentation, not validation.
class DeepAgentEvent(RootModel):
    """One SSE frame from GET /ask/agent/run/{run_id}: a `status` per tool call and a
    `result` per tool result as the agent works, then `sources` + a terminal `answer`
    — or `awaiting_approval`, or `error`."""

    root: Annotated[
        StatusEvent | ResultEvent | SourcesEvent | AnswerEvent | AwaitingApprovalEvent | ErrorEvent,
        Field(discriminator="type"),
    ]


# The Deep Agent path: answers from the corpus via its own tool-driven control
# flow. The run is traced (span lives in deepagent.run_deepagent).
@app.post("/ask/agent")
@limiter.limit(RATE_LIMIT)
def ask_deepagent(request: Request, body: DeepAgentRequest) -> DeepAgentResponse | AwaitingApproval:
    if check_input(body.question):
        return DeepAgentResponse(answer=BLOCKED, thread_id=body.thread_id)
    result = run_deepagent(body.question, body.thread_id, body.research_budget)
    return _agent_result(body.question, result)


def _agent_result(question: str, result: dict) -> DeepAgentResponse | AwaitingApproval:
    """Shared deep-agent response shaping: HITL pause passthrough, output rail on answers."""
    if result["status"] == "awaiting_approval":
        return AwaitingApproval(thread_id=result["thread_id"], pending=result["pending"])
    if check_output(question, result["answer"]):
        return DeepAgentResponse(answer=BLOCKED, thread_id=result["thread_id"])
    return DeepAgentResponse(answer=result["answer"], thread_id=result["thread_id"])


# Resume a paused thread: approve or reject the external-research gate. Works from a
# separate request even after a restart (state is durable in Postgres).
@app.post("/ask/agent/resume")
@limiter.limit(RATE_LIMIT)
def resume_deepagent_endpoint(
    request: Request, body: ResumeRequest
) -> DeepAgentResponse | AwaitingApproval:
    # No new question to input-check here; the resumed run's answer still gets the
    # output rail (empty question — self_check_output only reads the bot message).
    result = resume_deepagent(body.thread_id, body.decision)
    return _agent_result("", result)


# Deep-agent runs are decoupled from the HTTP connection: POST /ask/agent/run starts
# the agent in a background thread that buffers its events here, and GET /ask/agent/run/
# {run_id}?after=N replays buffered events then follows live. A mobile client whose
# connection dies while backgrounded reconnects with its event count and misses nothing;
# the run itself never stops.
# ponytail: in-memory, single-process; a restart loses live runs (finished answers
# still land in QA memory). Move to Redis/Postgres if the API ever runs >1 worker.
agent_runs: dict[str, dict] = {}


class AgentRunStarted(BaseModel):
    run_id: str


@app.post("/ask/agent/run")
@limiter.limit(RATE_LIMIT)
def start_deepagent_run(request: Request, body: DeepAgentRequest) -> AgentRunStarted:
    for run_id, run in list(agent_runs.items()):  # prune finished runs older than 1h
        if run["done"] and time.monotonic() - run["ended_at"] > 3600:
            agent_runs.pop(run_id, None)  # pop, not del: concurrent requests both prune

    run_id = uuid.uuid4().hex
    run = {"events": [], "done": False, "ended_at": None}
    agent_runs[run_id] = run

    def work():
        try:
            if check_input(body.question):
                run["events"].append({"type": "error", "message": BLOCKED})
                return
            for event in stream_deepagent(body.question, body.thread_id, body.research_budget):
                run["events"].append(event)
        except Exception as e:
            run["events"].append({"type": "error", "message": str(e)})
        finally:
            run["ended_at"] = time.monotonic()
            run["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return AgentRunStarted(run_id=run_id)


# SSE: replays the run's events from index `after`, then follows until the run ends.
@app.get(
    "/ask/agent/run/{run_id}",
    responses={200: {"model": DeepAgentEvent, "description": "SSE stream"}},
)
@limiter.limit(RATE_LIMIT)
def deepagent_run_events(request: Request, run_id: str, after: int = 0) -> EventSourceResponse:
    run = agent_runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run (finished long ago, or the server restarted)")

    def events():
        i = max(0, after)  # a negative cursor would index from the list's end
        while True:
            while i < len(run["events"]):
                yield run["events"][i]
                i += 1
            if run["done"]:
                return
            time.sleep(0.3)

    return sse(events())


# The chunk a citation points at, reconstructed in its place for the frontend.
@app.get("/source/{chunk_id}")
@limiter.limit(RATE_LIMIT)
def source(request: Request, chunk_id: int) -> SourcePassage:
    with connect() as conn:
        return SourcePassage(**source_passage(conn, chunk_id))


class QAMemory(BaseModel):
    key: str
    question: str
    created_at: datetime


class QAMemoryDetail(QAMemory):
    answer: str
    corpus_sources: list[CorpusSource]
    web_urls: list[str]
    research_files: dict[str, str]


# The deep agent's Q&A long-term memory (rag/query/memory.py pgvector cache), queried
# read-only.
@app.get("/qa")
@limiter.limit(RATE_LIMIT)
def qa_memories(request: Request) -> list[QAMemory]:
    with connect() as conn:
        return [QAMemory(**row) for row in list_memories(conn)]


@app.get("/qa/{key}")
@limiter.limit(RATE_LIMIT)
def qa_memory(request: Request, key: str) -> QAMemoryDetail:
    with connect() as conn:
        record = get_memory(conn, key)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return QAMemoryDetail(**record)
