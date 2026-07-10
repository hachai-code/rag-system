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

import psycopg
from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row

from ..config import CONFIG
from ..db import DB_URL, connect
from .answer import OPENROUTER_BASE_URL
from .retrieve import retrieve
from .web_search_agent import fetch_page as _fetch_page, search_web as _search_web

AGENT_MODEL = CONFIG.agent_model
RESEARCH_SUBAGENT_MODEL = CONFIG.research_subagent_model
AGENT_TOP_K = CONFIG.agent_top_k
AGENT_METHOD = CONFIG.agent_method
ENABLE_HITL = CONFIG.enable_hitl

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
you drew on. If nothing relevant turns up, say so."""


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
model = ChatOpenAI(
    model=AGENT_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=environ["OPENROUTER_API_KEY"],
    temperature=0,
    extra_body=_OPENROUTER_PROVIDER,
    stream_usage=True,
)

research_model = ChatOpenAI(
    model=RESEARCH_SUBAGENT_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=environ["OPENROUTER_API_KEY"],
    temperature=0,
    extra_body=_OPENROUTER_PROVIDER,
    stream_usage=True,
)

# One shared Langfuse handler, attached to each run's config so every LLM and tool
# call in the LangGraph run (across its worker threads) nests under the request's
# span, giving one trace per request with token usage. Reuses the singleton client,
# so it's a no-op when Langfuse keys are unset.
_langfuse_handler = CallbackHandler()


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


# Thin adapters over the web functions in web_search_agent.py, exposed to the
# research subagent as LangChain tools — the same seam the naked web agent uses.
@tool
def web_search(query: str) -> str:
    """Search the web; returns top results as title/url/snippet."""
    try:
        results = _search_web(query)
        return "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results) or "No results."
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


@tool
def fetch_page(url: str) -> str:
    """Fetch a URL and return its main text, boilerplate stripped."""
    try:
        return _fetch_page(url)
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


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
)


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


def run_deepagent(question: str, thread_id: str) -> dict:
    """Answer the question with the deep agent, traced as one Langfuse span.

    Returns `{"status": "done", "answer", "thread_id"}`, or — when the HITL gate
    pauses before external research — `{"status": "awaiting_approval", "thread_id",
    "pending"}`. Resume a paused thread with resume_deepagent()."""
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [_langfuse_handler]}
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent", input=question
    ) as span:
        result = AGENT.invoke({"messages": [{"role": "user", "content": question}]}, config)
        if result.get("__interrupt__"):
            pending = _pending_tool_calls(result["__interrupt__"])
            span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
            return {"status": "awaiting_approval", "thread_id": thread_id, "pending": pending}
        answer = result["structured_response"].answer
        span.update(output=answer, metadata={"thread_id": thread_id})
    return {"status": "done", "answer": answer, "thread_id": thread_id}


def resume_deepagent(thread_id: str, decision: str) -> dict:
    """Approve or reject the paused external-research gate on `thread_id` and continue
    the run — from a separate, later request, even across a restart (the state lives
    in Postgres). `decision` is "approve" or "reject"; one decision is sent per pending
    tool call. Returns the same shapes as run_deepagent (it may pause again)."""
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [_langfuse_handler]}
    pending = _pending_tool_calls(AGENT.get_state(config).interrupts)
    decisions = [{"type": decision} for _ in pending]
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent-resume", input=decision
    ) as span:
        result = AGENT.invoke(Command(resume={"decisions": decisions}), config)
        if result.get("__interrupt__"):
            span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
            return {
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "pending": _pending_tool_calls(result["__interrupt__"]),
            }
        answer = result["structured_response"].answer
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


def stream_deepagent(question: str, thread_id: str):
    """Yield event dicts as the deep agent works: a `status` event per tool call
    (with `call_id`, `tool`, `label`) and a `result` event per tool result (the
    preview, correlated by `call_id`), so the UI can show each step's call and its
    result. Subagent steps stream too (via subgraphs). Terminated by one `answer`
    — or `error` if the run blew up. Mirrors stream_agent()."""
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [_langfuse_handler]}
    _registries[thread_id] = {}  # fresh per-run registry retrieve_corpus fills in
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent", input=question
    ) as span:
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
                answer = state.values["structured_response"].answer
                span.update(output=answer, metadata={"thread_id": thread_id})
                yield {"type": "sources", "sources": _cited_corpus_sources(thread_id, answer)}
                yield {"type": "answer", "text": answer, "thread_id": thread_id}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            _registries.pop(thread_id, None)


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
