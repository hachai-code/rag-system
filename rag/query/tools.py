"""Shared toolset for the two-agent stack (rag/query/agent.py).

One set of tools both agents draw on:
- `web_search` (Tavily) + `fetch_page` (httpx + trafilatura, distilled when long) — the
  web-research tools, exposed to the web agent.
- `retrieve_corpus` — the corpus tool, wrapping the deterministic retrieve() funnel and
  rendering its `Hit`s as numbered, cite-able passages.

Tools are plain functions taking a Pydantic AI `RunContext`; per-run state (the web-call
budget, the corpus citation registry, seen URLs for citation checking) lives on the `deps`
object rather than the module-level dicts the old deep agent kept keyed by thread_id.

Importing this module needs no API keys or DB — clients build lazily on first tool call.
"""

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import httpx
import tiktoken
import trafilatura
from langfuse.openai import OpenAI
from pydantic import BaseModel
from pydantic_ai import RunContext

from ..clients import OPENROUTER_BASE_URL
from ..config import CONFIG
from ..db import Hit, connect
from .retrieve import retrieve

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_PAGE_TOKENS = 4000
DISTILL_MODEL = CONFIG.gen_models["flash"]
DISTILL_OVER_TOKENS = 1500  # pages shorter than this enter the transcript raw

AGENT_TOP_K = CONFIG.agent_top_k
AGENT_METHOD = CONFIG.agent_method

_encoder = tiktoken.get_encoding("o200k_base")

BUDGET_MSG = (
    "Research budget reached. Do not search or fetch further — write the final "
    "answer now from the findings you already have."
)


# --- Per-run state carried on Pydantic AI deps -------------------------------------


@dataclass
class Budget:
    """Per-run cap on web tool calls (search + fetch), shared across a whole deep-agent
    run (every research delegation charges the same counter). `limit` 0 means unlimited."""

    limit: int
    used: int = 0

    def charge(self) -> bool:
        """Charge one web call; return True once the budget is spent (tool should stop)."""
        if self.limit <= 0:
            return False
        self.used += 1
        return self.used > self.limit


@dataclass
class WebDeps:
    """Deps for the web agent: the shared budget and every URL seen in tool traffic
    (the citation output-validator checks the answer's citations against this set)."""

    budget: Budget
    seen_urls: set[str] = field(default_factory=set)


@dataclass
class CorpusDeps:
    """Deps for the corpus agent: the shared budget, the citation registry that keeps a
    passage's [n] stable across retrievals, and the similar-past-Q&As block to inject."""

    budget: Budget
    registry: dict[int, dict] = field(default_factory=dict)
    qa_block: str = ""


# --- URL helpers (citation verification) -------------------------------------------

# ")" only counts as part of a URL inside a balanced "(...)" pair, so Wikipedia-
# style URLs survive while markdown link closers and prose parens terminate.
_URL_RE = re.compile(r"https?://[^\s<>\"'()\]]*(?:\([^\s()]*\)[^\s<>\"'()\]]*)*")
_MD_LINK_RE = re.compile(r"\]\((https?://(?:[^()\s]|\([^()\s]*\))+)\)")


def _urls_in(text: str) -> set[str]:
    """All http(s) URLs in the text, normalized for comparison."""
    return {u.rstrip(".,;:").rstrip("/") for u in _URL_RE.findall(text)}


def _cited_urls(answer: str) -> set[str]:
    """URLs cited as markdown links — the citation format the prompt requires.
    Deliberately ignores bare URLs so placeholder URLs in code examples don't
    trigger false rejections."""
    return {u.rstrip(".,;:").rstrip("/") for u in _MD_LINK_RE.findall(answer)}


# --- Raw web functions -------------------------------------------------------------


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


def tavily_search(query: str, k: int = 5) -> list[SearchResult]:
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


def fetch_url(url: str) -> str:
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


@lru_cache(maxsize=1)
def _distill_client() -> OpenAI:
    """Langfuse-traced OpenAI client for _distill (raw chat.completions, not an agent)."""
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def _distill(client: OpenAI, question: str, page: str) -> tuple[str, float]:
    """Compress a fetched page to the parts relevant to the question, via the cheap
    flash model. On any failure returns the raw page — distillation is an optimization,
    never a point of failure."""
    prompt = (
        "You are compressing a fetched web page for a research agent.\n"
        f"Research question: {question}\n\n"
        "From the page text below, extract only content relevant to the question: "
        "facts, dates, numbers, names — quoting key wording verbatim where it "
        "matters. Max 300 words, no preamble. If nothing on the page is relevant, "
        "reply with one line saying what the page is about instead.\n\n"
        f"{page}"
    )
    try:
        resp = client.chat.completions.create(
            model=DISTILL_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return page, 0.0
        return text, getattr(resp.usage, "cost", None) or 0.0
    except Exception:
        return page, 0.0


# --- Corpus rendering --------------------------------------------------------------


def format_corpus_hits(hits: list[Hit], registry: dict[int, dict] | None = None) -> str:
    """Render retrieved chunks as numbered, cite-able passages for the corpus agent.

    Each hit becomes a `[n] Title (source)` header over its text, so the agent can cite a
    passage by its number and the reader can trace it back. Numbers are assigned by chunk
    id in `registry` and reused across calls, so a passage keeps its [n] however often it's
    retrieved. Without a registry, numbering is positional from 1."""
    if not hits:
        return "No relevant passages found in the corpus."
    if registry is None:
        registry = {}
    blocks = []
    for hit in hits:
        entry = registry.get(hit["id"])
        if entry is None:
            entry = {
                "n": len(registry) + 1,
                "chunk_id": hit["id"],
                "title": hit["title"],
                "source": hit["source"],
            }
            registry[hit["id"]] = entry
        blocks.append(f"[{entry['n']}] {hit['title']} ({hit['source']})\n{hit['content']}")
    return "\n\n".join(blocks)


# --- Tools (Pydantic AI function tools) --------------------------------------------


def web_search(ctx: RunContext[WebDeps], query: str) -> str:
    """Search the web via Tavily. Returns the top results as title/url/snippet. Queries:
    2-6 plain keywords, not full sentences. If results are off-target, search again with
    different terms."""
    if ctx.deps.budget.charge():
        return BUDGET_MSG
    try:
        results = tavily_search(query)
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"
    if not results:
        return f"No results found for {query!r}. Try a different query."
    for r in results:
        ctx.deps.seen_urls |= _urls_in(r.url) | _urls_in(r.snippet)
    return "\n\n".join(f"{r.title} ({r.url})\n{r.snippet}" for r in results)


def fetch_page(ctx: RunContext[WebDeps], url: str, focus: str) -> str:
    """Fetch a URL and return its main text. `focus` is the point you're researching;
    long pages are distilled down to just the parts relevant to it. Fails as text on dead
    links and paywalls — when that happens, pick a different source."""
    if ctx.deps.budget.charge():
        return BUDGET_MSG
    try:
        page = fetch_url(url)
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"
    ctx.deps.seen_urls |= _urls_in(url) | _urls_in(page)
    if len(_encoder.encode(page)) > DISTILL_OVER_TOKENS:
        page, _ = _distill(_distill_client(), focus, page)
    return page


def retrieve_corpus(ctx: RunContext[CorpusDeps], query: str) -> str:
    """Search the internal knowledge base for passages relevant to the query."""
    with connect() as conn:
        hits = retrieve(conn, query, k=AGENT_TOP_K, method=AGENT_METHOD)
    return format_corpus_hits(hits, ctx.deps.registry)
