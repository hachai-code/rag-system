"""The agentic /ask/agent path: a Deep Agent that answers from the corpus, then
enriches each point with external web research.

Unlike the deterministic /ask pipeline, the agent drives its own control flow —
it decides when to search the corpus (via the retrieve_corpus tool) and answers
from what it finds, using Deep Agents' built-in planning and virtual filesystem.
It then delegates each substantive point to a context-isolated external-research
subagent, so raw web crawl transcripts stay out of the main thread. The compiled
agent is built once at module load (like GRAPH in web_search_graph_agent.py) and
invoked per request inside a Langfuse span.

Model wiring: the deep agent needs a LangChain chat model. We reach DeepSeek
through OpenRouter (the same seam generation uses) via ChatOpenAI, reusing the
existing OPENROUTER_API_KEY rather than a separate DEEPSEEK_API_KEY.
"""

import re
from os import environ
from uuid import uuid4

import psycopg
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row

from ..config import CONFIG
from ..db import DB_URL, connect
from .answer import OPENROUTER_BASE_URL
from .retrieve import EMBED_DIM, VOYAGE_MODEL, _voyage, retrieve
from .web_search_agent import (
    DISTILL_OVER_TOKENS,
    _cited_urls,
    _distill,
    _encoder,
    fetch_page as _fetch_page,
    search_web as _search_web,
)

AGENT_MODEL = CONFIG.agent_model
RESEARCH_SUBAGENT_MODEL = CONFIG.research_subagent_model
AGENT_TOP_K = CONFIG.agent_top_k
AGENT_METHOD = CONFIG.agent_method
ENABLE_HITL = CONFIG.enable_hitl
RESEARCH_BUDGET = CONFIG.research_budget

AGENT_PROMPT = """You answer questions about the innerdance corpus, enriching each \
point with external web research.

Work in this order:
1. Search the internal knowledge base with the retrieve_corpus tool and draft an \
answer grounded in the passages it returns. Cite passages by their bracketed number, \
e.g. [1], [2].
2. Pull out the substantive points your draft rests on. Delegate each one to the \
external-research subagent (via the task tool) to corroborate or elaborate it with \
outside sources, and save the subagent's findings to a file named for that point.
3. Return the complete answer: every corpus point enriched with what the research \
turned up, written out in full and citing both the corpus passage [n] and the web \
URLs the subagent reported.

If the corpus does not cover the question, say so plainly instead of guessing."""

VALIDATION_PROMPT = """You research one specific point drawn from the innerdance \
corpus. Use web_search and fetch_page to find outside sources that corroborate, \
elaborate, or challenge it. Report what you found in a few sentences, citing the URLs \
you drew on. If nothing relevant turns up, say so.

Be economical: a couple of searches and a few page reads are enough. If a URL fails \
or errors, move on — do not retry variants of the same URL."""


# The agent's final answer is delivered through this typed field, not scraped from
# the last chat message — so bookkeeping (write_todos) and closing remarks can't
# displace it. Read as result["structured_response"].answer. ToolStrategy delivers
# it via a tool call (universally supported) rather than native json-schema, which
# some OpenRouter upstream providers reject.
class DeepAnswer(BaseModel):
    answer: str = Field(description="The complete answer, in full, with corpus [n] and web citations.")

# Route only to OpenRouter providers that support every parameter in our request
# (tool calling + structured output). Without this, when the preferred providers
# are rate-limited OpenRouter falls back to one that rejects the tool-heavy agent
# request with a 400 "invalid request params".
_OPENROUTER_PROVIDER = {
    "provider": {"require_parameters": True, "ignore": ["atlas-cloud"]}
}

# stream_usage=True so token usage rides the final streaming chunk (OpenRouter always
# returns it there) — otherwise the streamed run reports zero tokens to Langfuse.
# max_retries=6 (over the SDK's default 2) so a transient OpenRouter 5xx on any one of
# the run's ~100+ calls is retried (exponential backoff) instead of aborting the whole run.
model = ChatOpenAI(
    model=AGENT_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=environ["OPENROUTER_API_KEY"],
    temperature=0,
    extra_body=_OPENROUTER_PROVIDER,
    stream_usage=True,
    max_retries=6,
)

research_model = ChatOpenAI(
    model=RESEARCH_SUBAGENT_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=environ["OPENROUTER_API_KEY"],
    temperature=0,
    extra_body=_OPENROUTER_PROVIDER,
    stream_usage=True,
    max_retries=6,
)

# Raw Langfuse-traced OpenAI client for _distill (which calls chat.completions
# directly, not a LangChain model). Distillation itself runs on the cheap flash
# model wired inside _distill — not the subagent's model.
_distill_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=environ["OPENROUTER_API_KEY"])

# Per-run registries of corpus passages the agent has retrieved, keyed by thread_id:
# retrieve_corpus fills one so a passage keeps the same [n] across repeated retrievals
# and stream_deepagent can list the cited passages as sources. Keyed by thread_id
# (not a contextvar) because the tool runs in a different thread than the streaming
# generator. stream_deepagent resets and pops its entry each run.
# ponytail: module dict, one active run per thread_id (same as the checkpointer assumes).
_registries: dict[str, dict] = {}


def format_hits_for_deepagent(hits: list[dict], registry: dict | None = None) -> str:
    """Render retrieved chunks as numbered, cite-able passages for the deep agent.

    Each hit becomes a `[n] Title (source)` header over its text, so the agent can
    cite a passage by its number and the reader can trace it back to a document.
    Numbers are assigned by chunk id in `registry` and reused across calls, so a
    passage keeps its [n] no matter how often it's retrieved. Without a registry
    numbering is positional from 1."""
    if not hits:
        return "No relevant passages found in the corpus."
    if registry is None:
        registry = {}
    blocks = []
    for hit in hits:
        entry = registry.get(hit["id"])
        if entry is None:
            entry = {"n": len(registry) + 1, "chunk_id": hit["id"],
                     "title": hit["title"], "source": hit["source"]}
            registry[hit["id"]] = entry
        blocks.append(f"[{entry['n']}] {hit['title']} ({hit['source']})\n{hit['content']}")
    return "\n\n".join(blocks)


@tool
def retrieve_corpus(query: str, config: RunnableConfig) -> str:
    """Search the internal knowledge base for passages relevant to the query."""
    with connect() as conn:
        hits = retrieve(conn, query, k=AGENT_TOP_K, method=AGENT_METHOD)
    # `config` is injected by LangGraph (hidden from the model); thread_id keys the
    # registry so numbering is stable and stream_deepagent can read back the sources.
    thread_id = config.get("configurable", {}).get("thread_id", "")
    registry = _registries.setdefault(thread_id, {}) if thread_id else None
    return format_hits_for_deepagent(hits, registry)


# Per-run count of web tool calls (search + fetch), keyed by thread_id — seeded at 0
# at the start of each run and read via the config LangGraph injects into the tools.
# It caps a flailing research subagent (which can otherwise fetch dozens of dead URLs),
# bounding cost and the number of upstream calls that could hit a transient error.
# The budget defaults to RESEARCH_BUDGET but the caller can override it per run
# (_budgets); 0 means unlimited — the cap is off.
# ponytail: module dict, one active run per thread_id (same as the checkpointer assumes).
_web_calls: dict[str, int] = {}
_budgets: dict[str, int] = {}
_BUDGET_MSG = (
    "Research budget reached. Do not search or fetch further — write the final "
    "answer now from the findings you already have."
)


def _over_research_budget(config: RunnableConfig) -> bool:
    """Charge one web call against this run's budget; return True once it's spent."""
    thread_id = config.get("configurable", {}).get("thread_id", "")
    if not thread_id:
        return False
    budget = _budgets.get(thread_id, RESEARCH_BUDGET)
    if budget <= 0:  # 0 = unlimited
        return False
    _web_calls[thread_id] = _web_calls.get(thread_id, 0) + 1
    return _web_calls[thread_id] > budget


# Thin adapters over the web functions in web_search_agent.py, exposed to the
# research subagent as LangChain tools — the same seam the naked web agent uses.
@tool
def web_search(query: str, config: RunnableConfig) -> str:
    """Search the web; returns top results as title/url/snippet."""
    if _over_research_budget(config):
        return _BUDGET_MSG
    try:
        results = _search_web(query)
        return "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results) or "No results."
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


@tool
def fetch_page(url: str, focus: str, config: RunnableConfig) -> str:
    """Fetch a URL and return its main text. `focus` is the point you're researching;
    long pages are distilled down to just the parts relevant to it."""
    if _over_research_budget(config):
        return _BUDGET_MSG
    try:
        page = _fetch_page(url)
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"
    if len(_encoder.encode(page)) > DISTILL_OVER_TOKENS:
        page, _ = _distill(_distill_client, focus, page)
    return page


# Context-isolated: its web crawl transcript stays in the subagent thread and
# only its final findings return to the main agent (via the task tool). Given its
# own cheaper model through the "model" key, verified against deepagents 0.6.x.
research_subagent = {
    "name": "external-research",
    "description": "Search the web to corroborate or elaborate a single corpus point.",
    "system_prompt": VALIDATION_PROMPT,
    "tools": [web_search, fetch_page],
    "model": research_model,
}


# A single long-lived connection held for the process lifetime: the agent is
# compiled once at import, so PostgresSaver.from_conn_string (a context manager
# that closes on exit) won't do — we open the connection ourselves and keep it.
# autocommit + prepare_threshold=0 + dict_row mirror what from_conn_string sets.
_conn = psycopg.connect(
    DB_URL, autocommit=True, prepare_threshold=0, row_factory=dict_row
)
# Allowlist DeepAnswer so the checkpointed structured_response stays deserializable
# even under LANGGRAPH_STRICT_MSGPACK — otherwise resuming a thread would break.
_serde = JsonPlusSerializer(allowed_msgpack_modules=[("rag.query.deepagent", "DeepAnswer")])
_checkpointer = PostgresSaver(_conn, serde=_serde)
_checkpointer.setup()  # idempotent: creates checkpoint tables on first run

# Long-term memory store: cross-thread data, unlike the checkpointer's per-thread
# state. Its own connection (not _conn): saver and store each serialize access
# behind their own lock, so sharing one connection across concurrent checkpoint
# writes and store reads isn't safe.
_store_conn = psycopg.connect(
    DB_URL, autocommit=True, prepare_threshold=0, row_factory=dict_row
)


# PostgresStore calls embed_documents for BOTH put and search (there is no
# embed_query path), so one sync callable covers both sides. input_type="document"
# everywhere keeps stored questions and search queries in one consistent space —
# Voyage's query/document asymmetric tuning isn't reachable through this store.
def _embed_for_store(texts) -> list[list[float]]:
    return _voyage.embed(
        list(texts), model=VOYAGE_MODEL, input_type="document", output_dimension=EMBED_DIM
    ).embeddings


# Semantic index over the "question" field of qa records (score = cosine similarity,
# higher is better). setup() is idempotent per migration: the base store table is
# untouched; the index config adds the vector migrations (pgvector extension check,
# store_vectors table, HNSW index — CREATE INDEX CONCURRENTLY needs the autocommit
# connection above).
_store = PostgresStore(
    _store_conn,
    index={"dims": EMBED_DIM, "embed": _embed_for_store, "fields": ["question"]},
)
_store.setup()

_QA_NS = ("qa",)  # semantic Q&A cache — one shared namespace for the whole deployment

# DeepSeek flash can get stuck re-submitting an unchanged, fully-completed todo
# list instead of calling the DeepAnswer tool — and deepagents runs with
# recursion_limit=9999, so the loop burns thousands of calls before LangGraph
# steps in (observed live: "Updated todo list to ..." repeated indefinitely). A
# no-op write_todos never reaches the tool: the model gets told to answer instead.
class _TodoLoopBreaker(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if (call["name"] == "write_todos"
                and call["args"].get("todos") == request.state.get("todos")):
            return ToolMessage(
                content="Todo list unchanged — stop planning. Deliver the complete "
                        "final answer now by calling the DeepAnswer tool.",
                tool_call_id=call["id"],
            )
        return handler(request)


# The research subagent is dispatched through deepagents' built-in `task` tool, so
# to pause before external research we gate on `task` (not the subagent's name).
# Opt-in via config: the UI has no approval button yet, so defaulting HITL on would
# strand the streaming happy path at the interrupt.
_interrupt_on = (
    {"task": {"allowed_decisions": ["approve", "reject"]}} if ENABLE_HITL else None
)

AGENT = create_deep_agent(
    model=model,
    tools=[retrieve_corpus],
    system_prompt=AGENT_PROMPT,
    subagents=[research_subagent],
    checkpointer=_checkpointer,
    response_format=ToolStrategy(DeepAnswer),
    interrupt_on=_interrupt_on,
    middleware=[_TodoLoopBreaker()],
    store=_store,
)


_NO_ANSWER = "The agent finished without producing an answer."


def _final_answer(values: dict) -> str:
    """The typed answer when the model delivered one. The todo middleware tells the
    model to answer in plain text while ToolStrategy expects a DeepAnswer call — when
    the model follows the text route, salvage the prose instead of raising KeyError."""
    structured = values.get("structured_response")
    if structured is not None:
        return structured.answer
    for msg in reversed(values.get("messages", [])):
        if getattr(msg, "type", None) == "ai" and msg.text:
            return msg.text
    return _NO_ANSWER


def _save_qa_record(thread_id: str, question: str, answer: str, values: dict) -> None:
    """One Q&A cache record per completed run, semantically indexed on the question.
    put() embeds synchronously — one Voyage call after the answer already exists."""
    if answer == _NO_ANSWER:
        return
    files = values.get("files") or {}
    _store.put(_QA_NS, uuid4().hex, {
        "question": question,
        "answer": answer,
        "corpus_sources": _cited_corpus_sources(thread_id, answer),
        "web_urls": sorted(_cited_urls(answer)),
        # deepagents FileData -> plain text; skip base64 (binary) files
        "research_files": {
            path: fd["content"] for path, fd in files.items()
            if fd.get("encoding", "utf-8") == "utf-8"
        },
    })


def _pending_tool_calls(interrupts) -> list[dict]:
    """Flatten a batch of HITL interrupts into a JSON-serializable summary of the
    tool calls awaiting approval. A single gate can batch several `task` calls (the
    agent may delegate multiple research points at once), so this returns one entry
    per pending call — the count also drives how many decisions a resume must send."""
    pending = []
    for interrupt in interrupts:
        for action in interrupt.value["action_requests"]:
            args = action.get("args") or {}
            summary = args.get("description") or args.get("query") or ""
            pending.append({"tool": action["name"], "summary": " ".join(str(summary).split())[:200]})
    return pending


def run_deepagent(question: str, thread_id: str, research_budget: int | None = None) -> dict:
    """Answer the question with the deep agent, traced as one Langfuse span.

    `research_budget` caps web calls for this run (0 = unlimited); defaults to config.
    Returns `{"status": "done", "answer", "thread_id"}`, or — when the HITL gate
    pauses before external research — `{"status": "awaiting_approval", "thread_id",
    "pending"}`. Resume a paused thread with resume_deepagent()."""
    config = {"configurable": {"thread_id": thread_id}}
    _registries[thread_id] = {}  # fresh per-run registry retrieve_corpus fills in
    _web_calls[thread_id] = 0
    _budgets[thread_id] = RESEARCH_BUDGET if research_budget is None else research_budget
    try:
        with get_client().start_as_current_observation(
            as_type="span", name="rag-agent", input=question
        ) as span:
            # The handler must be built inside the active span so the LangGraph run nests
            # under it — a module-level handler binds to the trace root instead.
            config["callbacks"] = [CallbackHandler()]
            result = AGENT.invoke({"messages": [{"role": "user", "content": question}]}, config)
            if result.get("__interrupt__"):
                pending = _pending_tool_calls(result["__interrupt__"])
                span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
                return {"status": "awaiting_approval", "thread_id": thread_id, "pending": pending}
            answer = _final_answer(result)
            _save_qa_record(thread_id, question, answer, result)
            span.update(output=answer, metadata={"thread_id": thread_id})
        return {"status": "done", "answer": answer, "thread_id": thread_id}
    finally:
        _registries.pop(thread_id, None)
        _web_calls.pop(thread_id, None)
        _budgets.pop(thread_id, None)


def resume_deepagent(thread_id: str, decision: str) -> dict:
    """Approve or reject the paused external-research gate on `thread_id` and continue
    the run — from a separate, later request, even across a restart (the state lives
    in Postgres). `decision` is "approve" or "reject"; one decision is sent per pending
    tool call. Returns the same shapes as run_deepagent (it may pause again). Resumed
    runs are not written to the Q&A cache — the original question isn't in scope here."""
    config = {"configurable": {"thread_id": thread_id}}
    _web_calls[thread_id] = 0
    pending = _pending_tool_calls(AGENT.get_state(config).interrupts)
    decisions = [{"type": decision} for _ in pending]
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent-resume", input=decision
    ) as span:
        config["callbacks"] = [CallbackHandler()]  # inside the span so the run nests under it
        result = AGENT.invoke(Command(resume={"decisions": decisions}), config)
        if result.get("__interrupt__"):
            span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
            return {
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "pending": _pending_tool_calls(result["__interrupt__"]),
            }
        answer = _final_answer(result)
        span.update(output=answer, metadata={"thread_id": thread_id})
    return {"status": "done", "answer": answer, "thread_id": thread_id}


def _step_label(name: str, args: dict) -> str:
    """Human-readable summary for one tool call. Pure: (name, args) -> label."""
    if name == "retrieve_corpus":
        return f"Searching the corpus for “{args.get('query', '')}”"
    if name == "task":
        desc = " ".join(args.get("description", "").split())
        return f"Delegating research: {desc[:100]}…" if len(desc) > 100 else f"Delegating research: {desc}"
    if name == "web_search":
        return f"Web search: “{args.get('query', '')}”"
    if name == "fetch_page":
        return f"Reading {args.get('url', '')}"
    if name == "write_todos":
        return "Planning the research"
    return name


def _preview(content, limit: int = 600) -> str:
    """A short, readable snippet of a tool result for the collapsible trace step."""
    text = (content if isinstance(content, str) else str(content)).strip()
    return text[:limit] + "…" if len(text) > limit else text


def _cited_corpus_sources(thread_id: str, answer: str) -> list[dict]:
    """The corpus passages the answer actually cites, in [n] order. Reads the run's
    registry retrieve_corpus filled in and keeps only the numbers present in the
    answer text, so the reader sees exactly the sources behind the citations."""
    registry = _registries.get(thread_id, {})
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted((s for s in registry.values() if s["n"] in cited), key=lambda s: s["n"])


def stream_deepagent(question: str, thread_id: str, research_budget: int | None = None):
    """Yield event dicts as the deep agent works: a `status` event per tool call
    (with `call_id`, `tool`, `label`) and a `result` event per tool result (the
    preview, correlated by `call_id`), so the UI can show each step's call and its
    result. Subagent steps stream too (via subgraphs). Terminated by one `answer`
    — or `error` if the run blew up. `research_budget` caps web calls (0 = unlimited);
    defaults to config. Mirrors stream_agent()."""
    config = {"configurable": {"thread_id": thread_id}}
    _registries[thread_id] = {}  # fresh per-run registry retrieve_corpus fills in
    _web_calls[thread_id] = 0  # fresh research budget for this run
    _budgets[thread_id] = RESEARCH_BUDGET if research_budget is None else research_budget
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent", input=question
    ) as span:
        config["callbacks"] = [CallbackHandler()]  # inside the span so the run nests under it
        try:
            for namespace, update in AGENT.stream(
                {"messages": [{"role": "user", "content": question}]},
                config,
                stream_mode="updates",
                subgraphs=True,
            ):
                scope = "research" if namespace else "main"
                for delta in (update or {}).values():
                    messages = delta.get("messages", []) if isinstance(delta, dict) else []
                    for msg in messages:
                        if getattr(msg, "type", None) == "ai":
                            for call in msg.tool_calls or []:
                                args = call.get("args") or {}
                                yield {
                                    "type": "status",
                                    "scope": scope,
                                    "call_id": call.get("id", ""),
                                    "tool": call.get("name", ""),
                                    "label": _step_label(call.get("name", ""), args),
                                }
                        elif getattr(msg, "type", None) == "tool":
                            yield {
                                "type": "result",
                                "call_id": getattr(msg, "tool_call_id", ""),
                                "preview": _preview(msg.content),
                            }
            # The run either finished or paused at the HITL gate; the stream ends
            # either way, so read the terminal state and emit the matching event.
            state = AGENT.get_state(config)
            if state.interrupts:
                span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
                yield {
                    "type": "awaiting_approval",
                    "thread_id": thread_id,
                    "pending": _pending_tool_calls(state.interrupts),
                }
            else:
                # Read the final answer from the typed response channel, not the chat.
                answer = _final_answer(state.values)
                _save_qa_record(thread_id, question, answer, state.values)
                span.update(output=answer, metadata={"thread_id": thread_id})
                yield {"type": "sources", "sources": _cited_corpus_sources(thread_id, answer)}
                yield {"type": "answer", "text": answer, "thread_id": thread_id}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            _registries.pop(thread_id, None)
            _web_calls.pop(thread_id, None)
            _budgets.pop(thread_id, None)


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is innerdance?"
    print(f"Q: {question}\n")
    result = run_deepagent(question, thread_id="cli")
    if result["status"] == "awaiting_approval":
        print(f"Awaiting approval before research: {result['pending']}")
    else:
        print(result["answer"])
    get_client().flush()
