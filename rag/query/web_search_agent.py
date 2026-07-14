"""A web research agent as one readable file: a naked tool-call loop, no framework.

The model drives its own research — search the web via Tavily, read pages via
trafilatura — until it can answer, within hard budgets (iterations, dollars,
seconds). Three design rules:

- Tool failures never raise. Errors return to the model as text ("Tool error:
  ...", "No results found ...") so it can route around them instead of dying.
- Citations are mechanically verified. Every URL cited in the answer must have
  appeared in the run's tool traffic; an answer citing an unseen URL is
  rejected and the model rewrites it.
- Every step is traced to Langfuse: LLM generations (automatic, via the OpenAI
  wrapper), tool calls with inputs and outputs, and citation rejections.

Uses the same OpenRouter seam as complete() in answer.py.
"""

import json
import os
import re
import time
from datetime import date

import httpx
import tiktoken
import trafilatura
from langfuse import get_client
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field

from ..clients import OPENROUTER_BASE_URL
from ..config import CONFIG

# --- Config ------------------------------------------------------------------

MODEL = CONFIG.gen_model
MAX_TOKENS = CONFIG.max_tokens
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

MAX_ITERATIONS = 10
MAX_COST_USD = 0.50
MAX_SECONDS = 90
MAX_PAGE_TOKENS = 4000
MAX_CITATION_RETRIES = 2

DISTILL_MODEL = CONFIG.gen_models["flash"]
DISTILL_OVER_TOKENS = 1500  # pages shorter than this enter the transcript raw
# One review pass of the draft answer before returning. Measured on/off over 10
# eval questions (tuning-log.md): +1 correct — within run-to-run noise — for +30%
# latency and guaranteed time-limit exhaustion. Off by default at this budget.
SELF_CRITIQUE = False

_encoder = tiktoken.get_encoding("o200k_base")

# --- Prompts -------------------------------------------------------------------

SYSTEM_PROMPT = """Today is {today}. You are a research agent. You answer questions by searching the web and reading pages — never from memory alone.

Method:
1. Break the question into sub-questions and work through them one at a time.
2. For each sub-question, start with one broad search to survey what's out there.
3. Fetch the 1-2 most promising results and read them in full. Snippets are teasers, not sources — never answer from snippets alone.
4. Verify load-bearing facts (dates, versions, numbers, "latest"/"most" claims) against a second independent source before stating them. Once two sources agree, the fact is verified — never search for it again.
5. Refine and search again when results are off-target; a shorter, more specific query usually beats a longer one.

Stop when every part of the question is grounded in a page you actually fetched, or when further searching stops turning up anything new. Do not keep researching a sub-question you have already verified.

Tool failures (dead links, paywalls, empty results) come back as text. Treat them as information: pick a different source or rephrase the query — do not retry the same call and do not give up.

Answer requirements:
- Answer every part of the question in plain prose.
- Cite sources as markdown links [title](https://url) next to the claims they support. Only cite URLs from your search results or fetched pages — cited URLs are checked against your research trace and answers citing unseen URLs are rejected.
- Include every concrete detail you encountered that bears on the question — dates, numbers, names, titles, records, "firsts" — even secondary ones. A research answer errs on the side of completeness, not brevity.
- If something could not be verified, say so explicitly instead of guessing."""

CRITIQUE_PROMPT = """Before finalizing, review the draft answer you just wrote against the evidence actually gathered above (search results and fetched pages):
1. Is every part of the question answered?
2. Is every claim supported by a page you read or a search result you saw? Flag any claim you cannot point to evidence for — verify it or remove it.
3. Are any load-bearing facts (dates, numbers, names, "firsts") still unverified or missing?

If you find gaps, do the missing research now (search_web / fetch_page), then write the final answer. If the draft holds up, return it — improved where the review found weaknesses."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web via Tavily. Returns the top 5 results as title, URL, "
                "and a short snippet. Snippets are 1-3 sentences of page text — enough "
                "to judge relevance, not enough to answer from. Queries: 2-6 plain "
                'keywords ("python 3.13 release date"), not full sentences. The '
                "engine ignores operators like site: — don't use them. At most one "
                "quoted phrase per query. Add a year only when the question is about "
                "a specific year. If results are off-target, search again with "
                'different terms. Returns "No results found" for queries that match '
                "nothing — rephrase and retry."
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
                "Fetch a URL and return its content with navigation, ads, and "
                "boilerplate stripped. Short pages come back as full text; long pages "
                "are distilled to the facts relevant to the research question, with "
                "key wording quoted verbatim. Use this on the 1-2 most promising "
                "search results per search — reading the page is the only way to "
                "verify a snippet. Fails as text on dead links, paywalls (401/403), "
                "and pages with no extractable text; when that happens, pick a "
                "different source."
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

# --- Tools ---------------------------------------------------------------------


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
        return (f"Tool error: unknown tool {name!r}. It does not exist here — "
                "there is no todo list, plan, or subagent tool. Your only tools "
                "are search_web and fetch_page; continue the research with those.")
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


def _distill(client: OpenAI, question: str, page: str) -> tuple[str, float]:
    """Compress a fetched page to the parts relevant to the question, via the
    cheap flash model. Returns (text, cost). On any failure returns the raw page —
    distillation is an optimization, never a point of failure."""
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
            model=DISTILL_MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return page, 0.0
        return text, getattr(resp.usage, "cost", None) or 0.0
    except Exception:
        return page, 0.0


# --- State ---------------------------------------------------------------------

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


def _limit_hit(iterations: int, cost_spent: float, elapsed: float) -> str | None:
    """Name of the first exhausted budget, or None while within all three.
    Shared with the LangGraph rebuild (web_search_graph_agent.py)."""
    if iterations >= MAX_ITERATIONS:
        return "iterations"
    if cost_spent >= MAX_COST_USD:
        return "cost"
    if elapsed >= MAX_SECONDS:
        return "time"
    return None


def _rejected_citations(draft: str, sources: set[str],
                        citation_retries: int) -> tuple[list[str], str | None]:
    """([], None) when the draft's citations are acceptable (or retries are spent);
    otherwise the unseen URLs and the rewrite instruction to send back. Logs the
    rejection. Shared with the LangGraph rebuild."""
    unknown = sorted(_cited_urls(draft) - sources)
    if not unknown or citation_retries >= MAX_CITATION_RETRIES:
        return [], None
    print(f"!! rejected: cites unseen URLs {unknown}")
    get_client().create_event(name="citation-rejected", input=unknown)
    return unknown, (
        "Your answer cites URLs that never appeared in your "
        f"research: {unknown}. Every citation must be "
        "a URL from your search results or fetched pages. "
        "Rewrite the answer, removing or replacing those "
        "citations."
    )


class Step(BaseModel):
    """One executed tool call, as the model requested it."""

    tool: str
    args: str  # raw JSON string from the model
    result: str


class AgentState(BaseModel):
    """Everything a run knows about itself; run_agent reads and writes this
    instead of loose locals."""

    question: str
    steps: list[Step] = []
    sources: set[str] = set()  # every URL seen in tool traffic
    iterations: int = 0
    cost_spent: float = 0.0
    citation_retries: int = 0
    started: float = Field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def limit_hit(self) -> str | None:
        """Name of the first exhausted budget, or None while within all three."""
        return _limit_hit(self.iterations, self.cost_spent, self.elapsed)

    def record(self, step: Step) -> None:
        """Append the step and absorb its URLs into the run's known sources."""
        self.steps.append(step)
        self.sources |= _urls_in(step.args) | _urls_in(step.result)


# --- Loop ----------------------------------------------------------------------


def _best_effort_answer(client: OpenAI, model: str, messages: list, limit: str) -> str:
    """One bounded no-tools call: answer from the research gathered so far.
    Shared with the LangGraph rebuild; `model` is passed in because each agent
    module's MODEL is monkeypatched per eval run."""
    stop_msg = {
        "role": "user",
        "content": "Stop researching. Do not call any tools. Answer the original "
                   "question in plain prose, based only on what you have found so far.",
    }
    answer = ""
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=MAX_TOKENS, messages=messages + [stop_msg],
            )
        except Exception:
            break
        # DeepSeek sometimes leaks raw "<｜DSML｜tool_calls>" markup as text when
        # told to stop mid-research; everything from the special token on is garbage
        answer = (resp.choices[0].message.content or "").split("<｜")[0].strip()
        if answer:
            break
    if not answer:
        # salvage: the model's last substantive narration beats a shrug
        for m in reversed(messages):
            role = m["role"] if isinstance(m, dict) else m.role
            content = (m.get("content") if isinstance(m, dict) else m.content) or ""
            if role == "assistant" and content:
                answer = content.split("<｜")[0].strip()
                if answer:
                    break
    return (f"{answer or 'Could not produce an answer.'}"
            f"\n\n[Note: stopped early — {limit} limit reached]")


def run_agent(question: str) -> str:
    """Answer the question by letting the model drive research, within hard limits:
    MAX_ITERATIONS LLM calls, MAX_COST_USD spend, MAX_SECONDS wall time. On a limit
    the agent answers from what it has gathered so far instead of raising."""
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=MAX_SECONDS,
    )
    state = AgentState(question=question)
    critiqued = False
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
        {"role": "user", "content": question},
    ]
    with get_client().start_as_current_observation(
        as_type="span", name="web-search-agent", input=question
    ) as span:
        while True:
            limit = state.limit_hit()
            if limit:
                answer = _best_effort_answer(client, MODEL, messages, limit)
                break

            state.iterations += 1
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=messages
            )
            state.cost_spent += getattr(resp.usage, "cost", None) or 0.0
            msg = resp.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                if SELF_CRITIQUE and not critiqued:
                    critiqued = True
                    print("~> self-critique pass")
                    messages.append({"role": "user", "content": CRITIQUE_PROMPT})
                    continue
                _, rejection = _rejected_citations(
                    msg.content, state.sources, state.citation_retries)
                if rejection:
                    state.citation_retries += 1
                    messages.append({"role": "user", "content": rejection})
                    continue
                answer = msg.content
                break

            for call in msg.tool_calls:
                print(f"-> {call.function.name}({call.function.arguments})  "
                      f"[iter {state.iterations}, ${state.cost_spent:.4f}, {state.elapsed:.1f}s]")
                with get_client().start_as_current_observation(
                    as_type="tool", name=call.function.name, input=call.function.arguments
                ) as tool_obs:
                    result = _execute_tool(call.function.name, call.function.arguments)
                    if (call.function.name == "fetch_page"
                            and len(_encoder.encode(result)) > DISTILL_OVER_TOKENS):
                        result, distill_cost = _distill(client, state.question, result)
                        state.cost_spent += distill_cost
                    tool_obs.update(output=result)
                state.record(Step(tool=call.function.name,
                                  args=call.function.arguments, result=result))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        span.update(
            output=answer,
            metadata={"iterations": state.iterations,
                      "cost_usd": round(state.cost_spent, 4),
                      "limit_hit": limit,
                      "citation_retries": state.citation_retries,
                      "critiqued": critiqued},
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
