"""The agentic /ask/agent path: a Deep Agent that answers from the corpus.

Unlike the deterministic /ask pipeline, the agent drives its own control flow —
it decides when to search the corpus (via the retrieve_corpus tool) and answers
from what it finds, using Deep Agents' built-in planning and virtual filesystem.
The compiled agent is built once at module load (like GRAPH in
web_search_graph_agent.py) and invoked per request inside a Langfuse span.

Model wiring: the deep agent needs a LangChain chat model. We reach DeepSeek
through OpenRouter (the same seam generation uses) via ChatOpenAI, reusing the
existing OPENROUTER_API_KEY rather than a separate DEEPSEEK_API_KEY.
"""

from os import environ

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import get_client

from ..config import CONFIG
from ..db import connect
from .answer import OPENROUTER_BASE_URL
from .retrieve import retrieve

AGENT_MODEL = CONFIG.agent_model
AGENT_TOP_K = CONFIG.agent_top_k
AGENT_METHOD = CONFIG.agent_method

AGENT_PROMPT = """You answer questions about the innerdance corpus.

Search the internal knowledge base with the retrieve_corpus tool, then answer the \
question grounded in the passages it returns. Cite the passages you drew on by their \
bracketed number, e.g. [1], [2]. If the corpus does not cover the question, say so \
plainly instead of guessing."""

model = ChatOpenAI(
    model=AGENT_MODEL,
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


AGENT = create_deep_agent(
    model=model,
    tools=[retrieve_corpus],
    system_prompt=AGENT_PROMPT,
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
