"""The naked agent loop: a while loop in which the model either calls the one tool
(web search via Tavily) or returns a final answer; we execute the tool, append the
result, and repeat. No framework, same OpenRouter seam as complete() in answer.py."""

import json
import os

import httpx
from openai import OpenAI
from pydantic import BaseModel

from ..config import CONFIG

MODEL = CONFIG.gen_model
MAX_TOKENS = CONFIG.max_tokens
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You answer questions using web search. Use search_web to find current, relevant "
    "material before answering; search again with a different query if the results "
    "don't cover the question. Ground your answer in what the results say and "
    "mention the sources."
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
    }
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
            query = json.loads(call.function.arguments)["query"]
            print(f"-> search_web({query!r})")
            results = search_web(query)
            text = "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "Reached max iterations without a final answer."


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is the latest stable Python release?"
    print(f"Q: {question}\n")
    print(run_agent(question))
