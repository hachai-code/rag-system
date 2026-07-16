"""The agentic query tier: two Pydantic AI agents over one shared toolset (tools.py).

- `web_agent`: a research agent (web_search + fetch_page) whose output validator rejects
  answers citing URLs it never saw (raising ModelRetry to force a rewrite). Serves
  /agent/stream and the eval web baseline; also the research subagent below.
- `corpus_agent`: answers from the corpus (retrieve_corpus), delegating each substantive
  point to `web_agent` through the `research_point` tool (agent-as-tool → context
  isolation). Serves /ask/agent*. Its final answer is the typed `DeepAnswer`.

Replaces the previous three agent modules (the deep-agent harness, the graph agent, and
the hand-written web loop) and drops their four heavyweight orchestration dependencies.
Per-run state (web-call budget, corpus citation registry, seen URLs, similar-past-Q&As)
rides on Pydantic AI `deps`, not module dicts.

Durability/HITL: v2 HITL is stop-the-world (an approval-gated `research_point` ends the
run with a `DeferredToolRequests` output), so a pause serializes its message history to
the `agent_threads` table and POST /ask/agent/resume continues from it. HITL is opt-in
(config: enable_hitl).

Importing this module needs no keys or DB — the agents build lazily on first use.
"""

import asyncio
import re
from datetime import date
from functools import lru_cache

import psycopg
from langfuse import get_client
from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    Tool,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    capture_run_messages,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    ToolCallPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..clients import openrouter_model
from ..config import CONFIG
from ..db import connect
from .memory import lookup_similar_qa, save_qa_record
from .tools import (
    Budget,
    CorpusDeps,
    WebDeps,
    _cited_urls,
    fetch_page,
    retrieve_corpus,
    web_search,
)

# Module-level so evals/web_search/run_baseline.py can override them per run.
MODEL = CONFIG.gen_model
SELF_CRITIQUE = False  # retained for the eval baseline's flag; the review pass is off

AGENT_MODEL = CONFIG.agent_model
RESEARCH_SUBAGENT_MODEL = CONFIG.research_subagent_model
RESEARCH_BUDGET = CONFIG.research_budget
ENABLE_HITL = CONFIG.enable_hitl

MAX_TOKENS = CONFIG.max_tokens
DEEP_MAX_TOKENS = 8192  # a full DeepAnswer runs long; ceiling stops a mid-answer truncation
MAX_CITATION_RETRIES = 2  # output-validator retries before the last draft is accepted-as-is
WEB_REQUEST_LIMIT = 10  # model turns per web run before best-effort (was MAX_ITERATIONS)
CORPUS_REQUEST_LIMIT = 60  # model turns per deep-agent run (research budget is the real cap)

_NO_ANSWER = "The agent finished without producing an answer."
_STOP_MSG = (
    "Stop researching. Do not call any tools. Answer the original question in plain "
    "prose, based only on what you have found so far."
)

WEB_SYSTEM_PROMPT = """Today is {today}. You are a research agent. You answer questions by searching the web and reading pages — never from memory alone.

Method:
1. Break the question into sub-questions and work through them one at a time.
2. For each sub-question, start with one broad web_search to survey what's out there.
3. fetch_page the 1-2 most promising results and read them in full. Snippets are teasers, not sources — never answer from snippets alone.
4. Verify load-bearing facts (dates, versions, numbers, "latest"/"most" claims) against a second independent source before stating them. Once two sources agree, the fact is verified — never search for it again.
5. Refine and search again when results are off-target; a shorter, more specific query usually beats a longer one.

Stop when every part of the question is grounded in a page you actually fetched, or when further searching stops turning up anything new.

Tool failures (dead links, paywalls, empty results) come back as text. Treat them as information: pick a different source or rephrase the query — do not retry the same call and do not give up.

Answer requirements:
- Answer every part of the question in plain prose.
- Cite sources as markdown links [title](https://url) next to the claims they support. Only cite URLs from your search results or fetched pages — cited URLs are checked against your research trace and answers citing unseen URLs are rejected.
- Include every concrete detail you encountered that bears on the question — dates, numbers, names, titles, records, "firsts" — even secondary ones.
- If something could not be verified, say so explicitly instead of guessing."""

RESEARCH_TASK = """Research this one specific point drawn from the innerdance corpus. Use web_search and fetch_page to find outside sources that corroborate, elaborate, or challenge it. Report what you found in a few sentences, citing the URLs you drew on. If nothing relevant turns up, say so. Be economical: a couple of searches and a few page reads are enough."""

AGENT_PROMPT = """You answer questions about the innerdance corpus, enriching each point with external web research.

Work in this order:
1. Search the internal knowledge base with the retrieve_corpus tool and draft an answer grounded in the passages it returns. Cite passages by their bracketed number, e.g. [1], [2].
2. Pull out the substantive points your draft rests on. Delegate each one to the research_point tool to corroborate or elaborate it with outside sources.
3. Deliver the complete answer — every corpus point enriched with what the research turned up, written out in full, citing both the corpus passage [n] and the web URLs the research reported.

If the corpus does not cover the question, say so plainly instead of guessing.

You may be shown similar past Q&As from earlier runs under "Similar past Q&As". Treat them as leads, not truth: reuse their findings and sources when the past question genuinely matches, but re-verify against the corpus and redo web research when the match is loose. Never cite a past Q&A itself as a source."""


class DeepAnswer(BaseModel):
    """The corpus agent's typed final answer — delivered as structured output, not scraped
    from the last chat message, so it can't be displaced by tool bookkeeping."""

    answer: str = Field(
        description="The complete answer, in full, with corpus [n] and web citations."
    )


# --- Agent factories ---------------------------------------------------------------


@lru_cache(maxsize=1)
def web_agent() -> Agent[WebDeps, str]:
    """The web research agent. Tools cite URLs; the output validator rejects any citation
    that never appeared in the run's tool traffic (ModelRetry → rewrite)."""
    agent = Agent(
        openrouter_model(MODEL),
        deps_type=WebDeps,
        output_type=str,
        retries=MAX_CITATION_RETRIES,
        model_settings=ModelSettings(max_tokens=MAX_TOKENS),
        tools=[web_search, fetch_page],
    )

    @agent.instructions
    def _dated_prompt() -> str:  # per-run so the date can't go stale in the cached agent
        return WEB_SYSTEM_PROMPT.format(today=date.today().isoformat())

    @agent.output_validator
    def _verify_citations(ctx: RunContext[WebDeps], output: str) -> str:
        unknown = sorted(_cited_urls(output) - ctx.deps.seen_urls)
        if unknown:
            get_client().create_event(name="citation-rejected", input=unknown)
            raise ModelRetry(
                f"Your answer cites URLs that never appeared in your research: {unknown}. "
                "Every citation must be a URL from your search results or fetched pages. "
                "Rewrite the answer, removing or replacing those citations."
            )
        return output

    return agent


@lru_cache(maxsize=1)
def corpus_agent() -> Agent[CorpusDeps, DeepAnswer]:
    """The deep agent: answers from the corpus, delegating research to the web agent."""
    agent = Agent(
        openrouter_model(AGENT_MODEL),
        deps_type=CorpusDeps,
        output_type=DeepAnswer,
        model_settings=ModelSettings(max_tokens=DEEP_MAX_TOKENS),
        # Under HITL, research_point is approval-gated: a call pauses the run
        # (DeferredToolRequests) until resume_deepagent approves or denies it.
        tools=[retrieve_corpus, Tool(research_point, requires_approval=ENABLE_HITL)],
    )

    @agent.instructions
    def _prompt(ctx: RunContext[CorpusDeps]) -> str:
        if ctx.deps.qa_block:
            return f"{AGENT_PROMPT}\n\n## Similar past Q&As\n{ctx.deps.qa_block}"
        return AGENT_PROMPT

    return agent


async def research_point(ctx: RunContext[CorpusDeps], point: str) -> str:
    """Delegate one corpus point to the external web-research agent to corroborate or
    elaborate it. Returns its findings with cited URLs, in-band. One call per point."""
    try:
        return await _web_answer(
            f"{RESEARCH_TASK}\n\nPoint to research: {point}",
            WebDeps(budget=ctx.deps.budget),
            openrouter_model(RESEARCH_SUBAGENT_MODEL),
        )
    except Exception as e:  # tool failures never raise — report back as text
        return f"Research error: {type(e).__name__}: {e}"


# --- HITL wiring (approval-gated research_point) -----------------------------------


def _corpus_output():
    """Under HITL, an approval-gated research_point call ends the run with a
    DeferredToolRequests instead of a DeepAnswer, so it joins the output union."""
    return [DeepAnswer, DeferredToolRequests] if ENABLE_HITL else DeepAnswer


# --- Small helpers -----------------------------------------------------------------


def _args(part: ToolCallPart) -> dict:
    try:
        return part.args_as_dict()
    except Exception:
        return {}


def _step_label(name: str, args: dict) -> str:
    """Human-readable summary for one tool call. Pure: (name, args) -> label."""
    if name == "retrieve_corpus":
        return f"Searching the corpus for “{args.get('query', '')}”"
    if name == "research_point":
        point = " ".join(str(args.get("point", "")).split())
        return f"Delegating research: {point[:100]}…" if len(point) > 100 else f"Delegating research: {point}"
    if name == "web_search":
        return f"Web search: “{args.get('query', '')}”"
    if name == "fetch_page":
        return f"Reading {args.get('url', '')}"
    return name


def _preview(content, limit: int = 600) -> str:
    text = (content if isinstance(content, str) else str(content)).strip()
    return text[:limit] + "…" if len(text) > limit else text


def _final_answer(output) -> str:
    if isinstance(output, DeepAnswer):
        return output.answer or _NO_ANSWER
    if isinstance(output, str):
        return output or _NO_ANSWER
    return _NO_ANSWER


def _cited_corpus_sources(registry: dict, answer: str) -> list[dict]:
    """The corpus passages the answer actually cites, in [n] order."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted((s for s in registry.values() if s["n"] in cited), key=lambda s: s["n"])


def _pending(calls: list[ToolCallPart]) -> list[dict]:
    return [
        {"tool": c.tool_name, "summary": " ".join(str(_args(c).get("point", "")).split())[:200]}
        for c in calls
    ]


def _budget(research_budget: int | None) -> int:
    return RESEARCH_BUDGET if research_budget is None else research_budget


# --- Async ↔ sync bridge -----------------------------------------------------------


def _drive(agen):
    """Yield items from an async generator to sync callers (SSE routes, background runs)."""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        finally:
            loop.close()


# --- Web agent runs ----------------------------------------------------------------


async def _web_answer(prompt: str, deps: WebDeps, model) -> str:
    """Run the web agent within a request budget; on exhaustion, one bounded no-tools pass
    over the research gathered so far (best-effort) rather than raising."""
    limits = UsageLimits(request_limit=WEB_REQUEST_LIMIT)
    with capture_run_messages() as messages:
        try:
            result = await web_agent().run(prompt, deps=deps, model=model, usage_limits=limits)
            return result.output
        except (UsageLimitExceeded, UnexpectedModelBehavior):
            pass
    try:
        result = await web_agent().run(
            _STOP_MSG,
            message_history=messages,
            deps=deps,
            model=model,
            usage_limits=UsageLimits(request_limit=2),
        )
        return f"{result.output}\n\n[Note: stopped early — budget reached]"
    except Exception:
        return "Could not produce an answer.\n\n[Note: stopped early — budget reached]"


def run_agent(question: str) -> str:
    """The standalone web research agent (eval baseline). Answers within a request budget,
    reading MODEL at call time so run_baseline.py can override it."""
    deps = WebDeps(budget=Budget(limit=0))  # web-call cap off; the request budget bounds it
    with get_client().start_as_current_observation(
        as_type="span", name="web-search-agent", input=question
    ) as span:
        answer = asyncio.run(_web_answer(question, deps, openrouter_model(MODEL)))
        span.update(output=answer, metadata={"sources": len(deps.seen_urls)})
    return answer


async def _astream_web(question: str):
    deps = WebDeps(budget=Budget(limit=0))
    with get_client().start_as_current_observation(
        as_type="span", name="web-search-agent", input=question
    ) as span:
        try:
            async with web_agent().iter(
                question,
                deps=deps,
                model=openrouter_model(MODEL),
                usage_limits=UsageLimits(request_limit=WEB_REQUEST_LIMIT),
            ) as run:
                async for node in run:
                    if Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    p = event.part
                                    yield {
                                        "type": "tool_call",
                                        "name": p.tool_name,
                                        "id": p.tool_call_id,
                                        "arguments": p.args_as_json_str(),
                                    }
                                elif isinstance(event, FunctionToolResultEvent):
                                    r = event.result
                                    yield {
                                        "type": "tool_result",
                                        "id": r.tool_call_id,
                                        "preview": _preview(r.content),
                                    }
            answer = run.result.output
            span.update(output=answer)
            yield {"type": "done", "answer": answer, "sources": sorted(deps.seen_urls)}
        except Exception as e:
            yield {"type": "error", "message": str(e)}


def stream_agent(question: str):
    """SSE events for /agent/stream: tool_call / tool_result as the web agent works, then
    one terminal `done` (answer + sources) — or `error`."""
    yield from _drive(_astream_web(question))


# --- Deep agent runs ---------------------------------------------------------------


def _persist_thread(
    thread_id: str, question: str, messages, registry: dict, pending_ids: list[str]
) -> None:
    """Serialize a paused run's message history + citation registry + the pending
    approval-gated tool_call_ids to agent_threads, so resume can build DeferredToolResults."""
    blob = {
        "history": ModelMessagesTypeAdapter.dump_python(messages, mode="json"),
        "registry": {str(k): v for k, v in registry.items()},
        "pending_ids": pending_ids,
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO agent_threads (thread_id, question, messages, updated_at)"
            " VALUES (%s, %s, %s, now())"
            " ON CONFLICT (thread_id) DO UPDATE SET question = EXCLUDED.question,"
            " messages = EXCLUDED.messages, updated_at = now()",
            (thread_id, question, psycopg.types.json.Jsonb(blob)),
        )
        conn.commit()


def _load_thread(thread_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT question, messages FROM agent_threads WHERE thread_id = %s", (thread_id,)
        ).fetchone()
    if row is None:
        return None
    blob = row["messages"]
    return {
        "question": row["question"],
        "messages": ModelMessagesTypeAdapter.validate_python(blob["history"]),
        "registry": {int(k): v for k, v in blob["registry"].items()},
        "pending_ids": blob["pending_ids"],
    }


def _delete_thread(thread_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM agent_threads WHERE thread_id = %s", (thread_id,))
        conn.commit()


def _corpus_deps(research_budget: int | None, registry: dict | None = None) -> CorpusDeps:
    return CorpusDeps(budget=Budget(limit=_budget(research_budget)), registry=registry or {})


def run_deepagent(question: str, thread_id: str, research_budget: int | None = None) -> dict:
    """Answer the question with the deep agent, traced as one span. Returns `{"status":
    "done", "answer", "thread_id"}`, or — when HITL pauses before external research —
    `{"status": "awaiting_approval", "thread_id", "pending"}` (resume with resume_deepagent)."""
    deps = _corpus_deps(research_budget)
    with connect() as conn:
        deps.qa_block, top_score = lookup_similar_qa(conn, question)
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent", input=question
    ) as span:
        result = corpus_agent().run_sync(question, deps=deps, output_type=_corpus_output())
        output = result.output
        if isinstance(output, DeferredToolRequests):
            pending = output.approvals
            _persist_thread(
                thread_id, question, result.all_messages(), deps.registry,
                [c.tool_call_id for c in pending],
            )
            span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
            return {"status": "awaiting_approval", "thread_id": thread_id, "pending": _pending(pending)}
        answer = _final_answer(output)
        with connect() as conn:
            save_qa_record(
                conn, question, answer,
                _cited_corpus_sources(deps.registry, answer), sorted(_cited_urls(answer)), top_score,
            )
        span.update(output=answer, metadata={"thread_id": thread_id})
    return {"status": "done", "answer": answer, "thread_id": thread_id}


def resume_deepagent(thread_id: str, decision: str) -> dict:
    """Approve or reject the paused external-research gate on `thread_id` and continue —
    from a separate, later request, since the state lives in agent_threads. Resumed runs
    are not written to the Q&A cache (the original question isn't in scope here)."""
    loaded = _load_thread(thread_id)
    if loaded is None:
        return {"status": "done", "answer": _NO_ANSWER, "thread_id": thread_id}
    deps = _corpus_deps(None, registry=loaded["registry"])
    # Approve → the framework re-invokes research_point itself (using deps.budget); deny →
    # it returns a denial to the model automatically. No manual tool run either way.
    approved = decision == "approve"
    results = DeferredToolResults(approvals={cid: approved for cid in loaded["pending_ids"]})
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent-resume", input=decision
    ) as span:
        result = corpus_agent().run_sync(
            message_history=loaded["messages"], deferred_tool_results=results,
            deps=deps, output_type=_corpus_output(),
        )
        output = result.output
        if isinstance(output, DeferredToolRequests):  # another approval round
            pending = output.approvals
            _persist_thread(
                thread_id, loaded["question"], result.all_messages(), deps.registry,
                [c.tool_call_id for c in pending],
            )
            span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
            return {"status": "awaiting_approval", "thread_id": thread_id, "pending": _pending(pending)}
        answer = _final_answer(output)
        _delete_thread(thread_id)
        span.update(output=answer, metadata={"thread_id": thread_id})
    return {"status": "done", "answer": answer, "thread_id": thread_id}


async def _astream_deep(question: str, thread_id: str, research_budget: int | None):
    deps = _corpus_deps(research_budget)
    with connect() as conn:
        deps.qa_block, top_score = lookup_similar_qa(conn, question)
    with get_client().start_as_current_observation(
        as_type="span", name="rag-agent", input=question
    ) as span:
        try:
            async with corpus_agent().iter(
                question, deps=deps, output_type=_corpus_output()
            ) as run:
                async for node in run:
                    if Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    p = event.part
                                    yield {
                                        "type": "status",
                                        "scope": "main",
                                        "call_id": p.tool_call_id,
                                        "tool": p.tool_name,
                                        "label": _step_label(p.tool_name, _args(p)),
                                    }
                                elif isinstance(event, FunctionToolResultEvent):
                                    r = event.result
                                    yield {
                                        "type": "result",
                                        "call_id": r.tool_call_id,
                                        "preview": _preview(r.content),
                                    }
            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                pending = output.approvals
                _persist_thread(
                    thread_id, question, run.result.all_messages(), deps.registry,
                    [c.tool_call_id for c in pending],
                )
                span.update(output="awaiting_approval", metadata={"thread_id": thread_id})
                yield {"type": "awaiting_approval", "thread_id": thread_id, "pending": _pending(pending)}
                return
            answer = _final_answer(output)
            sources = _cited_corpus_sources(deps.registry, answer)
            with connect() as conn:
                save_qa_record(conn, question, answer, sources, sorted(_cited_urls(answer)), top_score)
            span.update(output=answer, metadata={"thread_id": thread_id})
            yield {"type": "sources", "sources": sources}
            yield {"type": "answer", "text": answer, "thread_id": thread_id}
        except Exception as e:
            yield {"type": "error", "message": str(e)}


def stream_deepagent(question: str, thread_id: str, research_budget: int | None = None):
    """Yield event dicts as the deep agent works: a `status` per tool call, a `result` per
    tool result, then a `sources` + `answer` (or `awaiting_approval`, or `error`)."""
    yield from _drive(_astream_deep(question, thread_id, research_budget))


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
