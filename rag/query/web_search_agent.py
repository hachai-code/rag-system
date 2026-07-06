"""The naked agent loop: a while loop in which the model either calls a tool
(web search via Tavily, page fetch via trafilatura) or returns a final answer;
we execute the tool, append the result, and repeat. No framework, same
OpenRouter seam as complete() in answer.py."""

import json
import os

import httpx
import tiktoken
import trafilatura
from openai import OpenAI
from pydantic import BaseModel

from ..config import CONFIG

MODEL = CONFIG.gen_model
MAX_TOKENS = CONFIG.max_tokens
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_ITERATIONS = 8
MAX_PAGE_TOKENS = 4000

_encoder = tiktoken.get_encoding("o200k_base")

SYSTEM_PROMPT = (
    "You answer questions by researching the web. Break multi-part questions into "
    "sub-questions and research them one at a time: use search_web to find candidate "
    "sources, fetch_page to read the most promising results in full, and search again "
    "with refined queries until every part is answered. Ground your answer in what "
    "the pages say and mention the sources."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web. Returns the most relevant results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a web page and return its main text. "
                           "Use on promising search results to read them in full.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The page URL."},
                },
                "required": ["url"],
            },
        },
    },
]


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


def search_web(query: str, k: int = 5) -> list[SearchResult]:
    """Return the k most relevant web results for the query, via Tavily."""
    resp = httpx.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json={"query": query, "max_results": k},
        timeout=30,
    )
    resp.raise_for_status()
    return [
        SearchResult(title=r["title"], url=r["url"], snippet=r["content"])
        for r in resp.json()["results"]
    ]


def fetch_page(url: str) -> str:
    """Return the main text of the page, truncated to MAX_PAGE_TOKENS."""
    resp = httpx.get(
        url,
        follow_redirects=True,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (research agent)"},
    )
    resp.raise_for_status()
    text = trafilatura.extract(resp.text, url=url)
    if text is None:
        return f"No extractable text at {url}."
    tokens = _encoder.encode(text)
    if len(tokens) > MAX_PAGE_TOKENS:
        text = _encoder.decode(tokens[:MAX_PAGE_TOKENS]) + "\n[truncated]"
    return text


def run_agent(question: str) -> str:
    """Answer the question by letting the model drive research: it decides when to
    search and when it has enough to answer."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    iterations = 0
    while iterations < MAX_ITERATIONS:
        iterations += 1
        msg = client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=messages
        ).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"-> {call.function.name}({args})")
            try:
                if call.function.name == "search_web":
                    results = search_web(args["query"])
                    text = "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results)
                else:
                    text = fetch_page(args["url"])
            except httpx.HTTPError as e:
                text = f"Tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "Reached max iterations without a final answer."


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Which was released more recently, the latest stable Python or the latest "
        "Node.js LTS, and what is one headline feature of each?"
    )
    print(f"Q: {question}\n")
    print(run_agent(question))
