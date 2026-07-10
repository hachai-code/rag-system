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

from os import environ

import psycopg
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langgraph.checkpoint.postgres import PostgresSaver
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

AGENT_PROMPT = """You answer questions about the innerdance corpus, enriching each \
point with external web research.

Work in this order:
1. Search the internal knowledge base with the retrieve_corpus tool and draft an \
answer grounded in the passages it returns. Cite passages by their bracketed number, \
e.g. [1], [2].
2. Pull out the substantive points your draft rests on. Delegate each one to the \
external-research subagent (via the task tool) to corroborate or elaborate it with \
outside sources, and save the subagent's findings to a file named for that point.
3. Present each corpus point enriched with what the research turned up, citing both \
the corpus passage [n] and the web URLs the subagent reported.

If the corpus does not cover the question, say so plainly instead of guessing."""

VALIDATION_PROMPT = """You research one specific point drawn from the innerdance \
corpus. Use web_search and fetch_page to find outside sources that corroborate, \
elaborate, or challenge it. Report what you found in a few sentences, citing the URLs \
you drew on. If nothing relevant turns up, say so."""

model = ChatOpenAI(
    model=AGENT_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=environ["OPENROUTER_API_KEY"],
    temperature=0,
)

research_model = ChatOpenAI(
    model=RESEARCH_SUBAGENT_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=environ["OPENROUTER_API_KEY"],
    temperature=0,
)


def format_hits_for_deepagent(hits: list[dict]) -> str:
    """Render retrieved chunks as numbered, cite-able passages for the deep agent.

    Each hit becomes a `[n] Title (source)` header over its text, so the agent can
    cite a passage by its number and the reader can trace it back to a document."""
    if not hits:
        return "No relevant passages found in the corpus."
    blocks = [
        f"[{i}] {hit['title']} ({hit['source']})\n{hit['content']}"
        for i, hit in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


@tool
def retrieve_corpus(query: str) -> str:
    """Search the internal knowledge base for passages relevant to the query."""
    with connect() as conn:
        hits = retrieve(conn, query, k=AGENT_TOP_K, method=AGENT_METHOD)
    return format_hits_for_deepagent(hits)


# Thin adapters over the web functions in web_search_agent.py, exposed to the
# research subagent as LangChain tools — the same seam the naked web agent uses.
@tool
def web_search(query: str) -> str:
    """Search the web; returns top results as title/url/snippet."""
    results = _search_web(query)
    return "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results) or "No results."


@tool
def fetch_page(url: str) -> str:
    """Fetch a URL and return its main text, boilerplate stripped."""
    return _fetch_page(url)


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
_checkpointer = PostgresSaver(_conn)
_checkpointer.setup()  # idempotent: creates checkpoint tables on first run

AGENT = create_deep_agent(
    model=model,
    tools=[retrieve_corpus],
    system_prompt=AGENT_PROMPT,
    subagents=[research_subagent],
    checkpointer=_checkpointer,
)


def run_deepagent(question: str, thread_id: str) -> dict:
    """Answer the question with the deep agent, traced as one Langfuse span."""
    config = {"configurable": {"thread_id": thread_id}}
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent", input=question
    ) as span:
        result = AGENT.invoke({"messages": [{"role": "user", "content": question}]}, config)
        answer = result["messages"][-1].content
        span.update(output=answer, metadata={"thread_id": thread_id})
    return {"answer": answer, "thread_id": thread_id}


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is innerdance?"
    print(f"Q: {question}\n")
    print(run_deepagent(question, thread_id="cli")["answer"])
    get_client().flush()
