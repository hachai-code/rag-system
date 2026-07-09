"""The web research agent from web_search_agent.py, rebuilt on LangGraph.

Same behavior, budgets, and Langfuse trace shape — but the hand-written while
loop is a StateGraph: call_model → run_tools (tool calls) or review (draft
answer), with every path checked against the budgets and diverted to
best_effort when one is exhausted. Shares prompts, tools, and helpers with the
old module during the A/B period; the old file goes away once the graph wins
the eval.
"""

import operator
import time
from datetime import date
from functools import cache
from os import environ
from typing import Annotated, TypedDict

from langfuse import get_client
from langfuse.openai import OpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import CONFIG
from .web_search_agent import (
    CRITIQUE_PROMPT,
    DISTILL_OVER_TOKENS,
    MAX_CITATION_RETRIES,
    MAX_COST_USD,
    MAX_ITERATIONS,
    MAX_SECONDS,
    MAX_TOKENS,
    OPENROUTER_BASE_URL,
    SYSTEM_PROMPT,
    TOOLS,
    _cited_urls,
    _distill,
    _encoder,
    _execute_tool,
    _urls_in,
)

# Module-level like the old agent: run_baseline.py patches these per run.
MODEL = CONFIG.gen_model
SELF_CRITIQUE = False


@cache
def _client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=environ["OPENROUTER_API_KEY"],
        timeout=MAX_SECONDS,
    )


# --- State ---------------------------------------------------------------------


class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]  # OpenAI-format message dicts, appended
    sources: set[str]  # every URL seen in tool traffic
    iterations: int
    cost_spent: float
    citation_retries: int
    critiqued: bool
    started: float  # time.monotonic() at run start
    limit: str | None  # budget that ended the run, None on a normal finish
    answer: str


def limit_hit(state: AgentState) -> str | None:
    """Name of the first exhausted budget, or None while within all three."""
    if state["iterations"] >= MAX_ITERATIONS:
        return "iterations"
    if state["cost_spent"] >= MAX_COST_USD:
        return "cost"
    if time.monotonic() - state["started"] >= MAX_SECONDS:
        return "time"
    return None


# --- Nodes ---------------------------------------------------------------------


def call_model(state: AgentState) -> dict:
    resp = _client().chat.completions.create(
        model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=state["messages"]
    )
    # plain dict, not the SDK object: keeps the state checkpoint-serializable
    return {
        "messages": [resp.choices[0].message.model_dump(exclude_none=True)],
        "iterations": state["iterations"] + 1,
        "cost_spent": state["cost_spent"] + (getattr(resp.usage, "cost", None) or 0.0),
    }


def run_tools(state: AgentState) -> dict:
    """Execute every tool call in the last assistant message."""
    sources = set(state["sources"])
    cost = state["cost_spent"]
    tool_messages = []
    for call in state["messages"][-1]["tool_calls"]:
        name, arguments = call["function"]["name"], call["function"]["arguments"]
        print(f"-> {name}({arguments})  "
              f"[iter {state['iterations']}, ${cost:.4f}, "
              f"{time.monotonic() - state['started']:.1f}s]")
        with get_client().start_as_current_observation(
            as_type="tool", name=name, input=arguments
        ) as tool_obs:
            result = _execute_tool(name, arguments)
            if (name == "fetch_page"
                    and len(_encoder.encode(result)) > DISTILL_OVER_TOKENS):
                result, distill_cost = _distill(_client(), state["question"], result)
                cost += distill_cost
            tool_obs.update(output=result)
        sources |= _urls_in(arguments) | _urls_in(result)
        tool_messages.append(
            {"role": "tool", "tool_call_id": call["id"], "content": result}
        )
    return {"messages": tool_messages, "sources": sources, "cost_spent": cost}


def review(state: AgentState) -> dict:
    """Gatekeep a draft answer: critique once if enabled, reject unseen
    citations, otherwise accept."""
    draft = state["messages"][-1].get("content") or ""
    if SELF_CRITIQUE and not state["critiqued"]:
        print("~> self-critique pass")
        return {
            "messages": [{"role": "user", "content": CRITIQUE_PROMPT}],
            "critiqued": True,
        }
    unknown = _cited_urls(draft) - state["sources"]
    if unknown and state["citation_retries"] < MAX_CITATION_RETRIES:
        print(f"!! rejected: cites unseen URLs {sorted(unknown)}")
        get_client().create_event(name="citation-rejected", input=sorted(unknown))
        return {
            "messages": [{
                "role": "user",
                "content": "Your answer cites URLs that never appeared in your "
                           f"research: {sorted(unknown)}. Every citation must be "
                           "a URL from your search results or fetched pages. "
                           "Rewrite the answer, removing or replacing those "
                           "citations.",
            }],
            "citation_retries": state["citation_retries"] + 1,
        }
    return {"answer": draft}


def best_effort(state: AgentState) -> dict:
    """One bounded no-tools call: answer from the research gathered so far."""
    limit = limit_hit(state) or "unknown"
    stop_msg = {
        "role": "user",
        "content": "Stop researching. Do not call any tools. Answer the original "
                   "question in plain prose, based only on what you have found so far.",
    }
    answer = ""
    for _ in range(3):
        try:
            resp = _client().chat.completions.create(
                model=MODEL, max_tokens=MAX_TOKENS,
                messages=state["messages"] + [stop_msg],
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
        for m in reversed(state["messages"]):
            if m["role"] == "assistant" and m.get("content"):
                answer = m["content"].split("<｜")[0].strip()
                if answer:
                    break
    return {
        "answer": (f"{answer or 'Could not produce an answer.'}"
                   f"\n\n[Note: stopped early — {limit} limit reached]"),
        "limit": limit,
    }


# --- Graph ---------------------------------------------------------------------


def route_response(state: AgentState) -> str:
    """After call_model: run requested tools, or review the draft answer."""
    return "run_tools" if state["messages"][-1].get("tool_calls") else "review"


def continue_or_stop(state: AgentState) -> str:
    """After run_tools or review: done, out of budget, or another model turn."""
    if state["answer"]:
        return END
    return "best_effort" if limit_hit(state) else "call_model"


_builder = StateGraph(AgentState)
_builder.add_node(call_model)
_builder.add_node(run_tools)
_builder.add_node(review)
_builder.add_node(best_effort)
_builder.add_edge(START, "call_model")
_builder.add_conditional_edges("call_model", route_response)
_builder.add_conditional_edges("run_tools", continue_or_stop)
_builder.add_conditional_edges("review", continue_or_stop)
_builder.add_edge("best_effort", END)
GRAPH = _builder.compile()


def _initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "messages": [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
            {"role": "user", "content": question},
        ],
        "sources": set(),
        "iterations": 0,
        "cost_spent": 0.0,
        "citation_retries": 0,
        "critiqued": False,
        "started": time.monotonic(),
        "limit": None,
        "answer": "",
    }


def run_agent(question: str) -> str:
    """Answer the question by letting the model drive research, within hard limits:
    MAX_ITERATIONS LLM calls, MAX_COST_USD spend, MAX_SECONDS wall time. On a limit
    the agent answers from what it has gathered so far instead of raising."""
    state = _initial_state(question)
    with get_client().start_as_current_observation(
        as_type="span", name="web-search-agent", input=question
    ) as span:
        # Default recursion_limit is 25 supersteps; a full run needs ~3 per
        # iteration (call_model, run_tools, review).
        final = GRAPH.invoke(state, config={"recursion_limit": 3 * MAX_ITERATIONS + 10})
        span.update(
            output=final["answer"],
            metadata={"iterations": final["iterations"],
                      "cost_usd": round(final["cost_spent"], 4),
                      "limit_hit": final["limit"],
                      "citation_retries": final["citation_retries"],
                      "critiqued": final["critiqued"]},
        )
    return final["answer"]


def run_hitl(question: str) -> str:
    """Human-in-the-loop demo: same graph, but compiled with a checkpointer and
    a static interrupt, so it pauses for approval before every tool batch.
    Untraced CLI exercise, not a production path."""
    graph = _builder.compile(
        checkpointer=InMemorySaver(), interrupt_before=["run_tools"]
    )
    config = {"configurable": {"thread_id": "cli"},
              "recursion_limit": 3 * MAX_ITERATIONS + 10}
    state = graph.invoke(_initial_state(question), config)
    while graph.get_state(config).next:  # paused before run_tools
        for call in state["messages"][-1]["tool_calls"]:
            print(f"?? wants {call['function']['name']}({call['function']['arguments']})")
        if input("approve? [Y/n] ").strip().lower() == "n":
            return "Aborted at human review."
        state = graph.invoke(None, config)  # resume from the checkpoint
    return state["answer"]


if __name__ == "__main__":
    import sys

    hitl = "--hitl" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--hitl"]
    question = args[0] if args else (
        "Which was released more recently, the latest stable Python or the latest "
        "Node.js LTS, and what is one headline feature of each?"
    )
    print(f"Q: {question}\n")
    print(run_hitl(question) if hitl else run_agent(question))
    get_client().flush()
