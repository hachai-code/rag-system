"""The naked agent loop: a while loop in which the model either calls a tool
(web search via Tavily, page fetch via trafilatura) or returns a final answer;
we execute the tool, append the result, and repeat. No framework, same
OpenRouter seam as complete() in answer.py."""

import json
import os
import re
import time

import httpx
import tiktoken
import trafilatura
from langfuse import get_client
from langfuse.openai import OpenAI
from pydantic import BaseModel

from ..config import CONFIG

MODEL = CONFIG.gen_model
MAX_TOKENS = CONFIG.max_tokens
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_ITERATIONS = 10
MAX_COST_USD = 0.50
MAX_SECONDS = 60
MAX_PAGE_TOKENS = 4000
MAX_CITATION_RETRIES = 2

_encoder = tiktoken.get_encoding("o200k_base")

# ponytail: clips URLs containing ")" (some Wikipedia pages); fine for this agent
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


_MD_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


def _urls_in(text: str) -> set[str]:
    """All http(s) URLs in the text, normalized for comparison."""
    return {u.rstrip(".,;:").rstrip("/") for u in _URL_RE.findall(text)}


def _cited_urls(answer: str) -> set[str]:
    """URLs cited as markdown links — the citation format the prompt requires.
    Deliberately ignores bare URLs so placeholder URLs in code examples don't
    trigger false rejections."""
    return {u.rstrip(".,;:").rstrip("/") for u in _MD_LINK_RE.findall(answer)}

SYSTEM_PROMPT = """You are a research agent. You answer questions by searching the web and reading pages — never from memory alone.

Method:
1. Break the question into sub-questions and work through them one at a time.
2. For each sub-question, start with one broad search to survey what's out there.
3. Fetch the 1-2 most promising results and read them in full. Snippets are teasers, not sources — never answer from snippets alone.
4. Verify load-bearing facts (dates, versions, numbers, "latest"/"most" claims) against a second independent source before stating them.
5. Refine and search again when results are off-target; a shorter, more specific query usually beats a longer one.

Stop when every part of the question is grounded in a page you actually fetched, or when further searching stops turning up anything new. Do not keep researching a sub-question you have already verified.

Tool failures (dead links, paywalls, empty results) come back as text. Treat them as information: pick a different source or rephrase the query — do not retry the same call and do not give up.

Answer requirements:
- Answer every part of the question in plain prose.
- Cite sources as markdown links [title](https://url) next to the claims they support. Only cite URLs from your search results or fetched pages — cited URLs are checked against your research trace and answers citing unseen URLs are rejected.
- If something could not be verified, say so explicitly instead of guessing."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web via Tavily. Returns the top 5 results as title, URL, "
                "and a short snippet. Snippets are 1-3 sentences of page text — enough "
                "to judge relevance, not enough to answer from. Use short, specific "
                'keyword queries ("python 3.13 release date") rather than full '
                "sentences. If results are off-target, search again with different "
                'terms. Returns "No results found" for queries that match nothing — '
                "rephrase and retry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": 'Search query. Short and specific works best, '
                                       'e.g. "node.js LTS latest version" not "what is '
                                       'the latest LTS version of node.js released".',
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch a URL and return the page's main text with navigation, ads, "
                "and boilerplate stripped. Long pages are cut at ~4000 tokens and end "
                'with "[truncated]". Use this on the 1-2 most promising search results '
                "per search — reading the full page is the only way to verify a "
                "snippet. Fails as text on dead links, paywalls (401/403), and pages "
                "with no extractable text; when that happens, pick a different source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL, usually taken from a search_web result.",
                    },
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


def _execute_tool(name: str, arguments: str) -> str:
    """Run one tool call; any failure comes back as text so the model can react."""
    try:
        args = json.loads(arguments)
        if name == "search_web":
            results = search_web(args["query"])
            if not results:
                return f"No results found for {args['query']!r}. Try a different query."
            return "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results)
        if name == "fetch_page":
            return fetch_page(args["url"])
        return f"Tool error: unknown tool {name!r}."
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


def _best_effort_answer(client, messages: list, limit: str) -> str:
    """One bounded no-tools call: answer from the research gathered so far."""
    stop_msg = {
        "role": "user",
        "content": "Stop researching. Do not call any tools. Answer the original "
                   "question in plain prose, based only on what you have found so far.",
    }
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, tool_choice="none",
            messages=messages + [stop_msg],
        )
        answer = resp.choices[0].message.content or ""
    except Exception:
        answer = "Could not produce an answer."
    return f"{answer}\n\n[Note: stopped early — {limit} limit reached]"


def run_agent(question: str) -> str:
    """Answer the question by letting the model drive research, within hard limits:
    MAX_ITERATIONS LLM calls, MAX_COST_USD spend, MAX_SECONDS wall time. On a limit
    the agent answers from what it has gathered so far instead of raising."""
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=MAX_SECONDS,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    start = time.monotonic()
    cost = 0.0
    iterations = 0
    citation_retries = 0
    seen_urls: set[str] = set()
    with get_client().start_as_current_observation(
        as_type="span", name="web-search-agent", input=question
    ) as span:
        while True:
            limit = ("iterations" if iterations >= MAX_ITERATIONS
                     else "cost" if cost >= MAX_COST_USD
                     else "time" if time.monotonic() - start >= MAX_SECONDS
                     else None)
            if limit:
                answer = _best_effort_answer(client, messages, limit)
                break
            iterations += 1
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=messages
            )
            cost += getattr(resp.usage, "cost", None) or 0.0
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                unknown = _cited_urls(msg.content) - seen_urls
                if unknown and citation_retries < MAX_CITATION_RETRIES:
                    citation_retries += 1
                    print(f"!! rejected: cites unseen URLs {sorted(unknown)}")
                    messages.append({
                        "role": "user",
                        "content": "Your answer cites URLs that never appeared in your "
                                   f"research: {sorted(unknown)}. Every citation must be "
                                   "a URL from your search results or fetched pages. "
                                   "Rewrite the answer, removing or replacing those "
                                   "citations.",
                    })
                    continue
                answer = msg.content
                break
            for call in msg.tool_calls:
                print(f"-> {call.function.name}({call.function.arguments})  "
                      f"[iter {iterations}, ${cost:.4f}, {time.monotonic() - start:.1f}s]")
                text = _execute_tool(call.function.name, call.function.arguments)
                seen_urls |= _urls_in(call.function.arguments) | _urls_in(text)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        span.update(
            output=answer,
            metadata={"iterations": iterations, "cost_usd": round(cost, 4),
                      "limit_hit": limit, "citation_retries": citation_retries},
        )
    return answer


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Which was released more recently, the latest stable Python or the latest "
        "Node.js LTS, and what is one headline feature of each?"
    )
    print(f"Q: {question}\n")
    print(run_agent(question))
    get_client().flush()
