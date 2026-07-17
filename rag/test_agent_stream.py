"""The two streaming agent paths, end to end: what stream_agent / stream_deepagent emit.

Runs the real agents, tools, deps threading and event loop bridge, with TestModel standing
in for the model and the key/network/DB seams stubbed — so it needs no API keys and no
database. The assertions pin the SSE event dicts the frontend reads.

Run: uv run pytest rag/test_agent_stream.py
"""

import json
from unittest.mock import MagicMock

import pytest
from pydantic_ai.models.test import TestModel

from rag.query import agent as agent_module
from rag.query import tools as tools_module

HIT = {
    "id": 7,
    "title": "Innerdance 101",
    "source": "book.pdf",
    "content": "Innerdance is a somatic practice.",
    "distance": 0.1,
}

SEARCH_RESULT = tools_module.SearchResult(
    title="Innerdance",
    url="https://example.com/innerdance",
    snippet="A somatic practice.",
)


@pytest.fixture
def stub_seams(monkeypatch):
    """Cut every seam that would reach a key, the network or the DB. The cached agent
    factories are emptied around the test so they build against the TestModel the test
    patches in, and no TestModel-backed agent outlives it."""
    monkeypatch.setattr(agent_module, "get_client", MagicMock())  # langfuse spans
    monkeypatch.setattr(agent_module, "connect", MagicMock())
    monkeypatch.setattr(tools_module, "connect", MagicMock())
    monkeypatch.setattr(tools_module, "tavily_search", lambda query, k=5: [SEARCH_RESULT])
    monkeypatch.setattr(tools_module, "retrieve", lambda conn, query, k, method: [HIT])
    monkeypatch.setattr(agent_module, "lookup_similar_qa", lambda conn, question: ("", 0.0))
    agent_module.web_agent.cache_clear()
    agent_module.corpus_agent.cache_clear()
    yield
    agent_module.web_agent.cache_clear()
    agent_module.corpus_agent.cache_clear()


def test_web_stream_emits_tool_traffic_then_done(monkeypatch, stub_seams):
    model = TestModel(call_tools=["web_search"], custom_output_text="Innerdance is a practice.")
    monkeypatch.setattr(agent_module, "openrouter_model", lambda _model_id: model)

    events = list(agent_module.stream_agent("what is innerdance?"))

    assert [e["type"] for e in events] == ["tool_call", "tool_result", "done"]
    call, result, done = events
    assert call["name"] == "web_search"
    assert "query" in json.loads(call["arguments"])
    assert result["id"] == call["id"]
    assert "https://example.com/innerdance" in result["preview"]
    assert done["answer"] == "Innerdance is a practice."
    # The URLs the tool saw, collected on deps through the run.
    assert done["sources"] == ["https://example.com/innerdance"]


def test_deep_stream_emits_status_result_then_sources_and_answer(monkeypatch, stub_seams):
    model = TestModel(
        call_tools=["retrieve_corpus"],
        custom_output_args={"answer": "Innerdance is a somatic practice [1]."},
    )
    monkeypatch.setattr(agent_module, "openrouter_model", lambda _model_id: model)
    saved = []
    monkeypatch.setattr(agent_module, "save_qa_record", lambda *args: saved.append(args))

    events = list(agent_module.stream_deepagent("what is innerdance?", "t1"))

    assert [e["type"] for e in events] == ["status", "result", "sources", "answer"]
    status, result, sources, answer = events
    assert status["scope"] == "main"
    assert status["tool"] == "retrieve_corpus"
    assert status["label"].startswith("Searching the corpus for")
    assert result["call_id"] == status["call_id"]
    assert "[1] Innerdance 101 (book.pdf)" in result["preview"]
    # [1] in the answer resolves through the citation registry the tool filled.
    assert sources["sources"] == [
        {"n": 1, "chunk_id": 7, "title": "Innerdance 101", "source": "book.pdf"}
    ]
    assert answer["text"] == "Innerdance is a somatic practice [1]."
    assert answer["thread_id"] == "t1"
    assert saved, "the finished run writes its answer to the Q&A cache"


def test_deep_stream_pauses_on_the_hitl_gate(monkeypatch, stub_seams):
    monkeypatch.setattr(agent_module, "ENABLE_HITL", True)  # opt-in in config; off by default
    model = TestModel(call_tools=["research_point"])
    monkeypatch.setattr(agent_module, "openrouter_model", lambda _model_id: model)

    events = list(agent_module.stream_deepagent("what is innerdance?", "t1"))

    # An approval-gated research_point ends the run with a DeferredToolRequests instead of
    # an answer; _persist_thread serializes the history for real (its DB write is stubbed).
    assert events[-1] == {
        "type": "awaiting_approval",
        "thread_id": "t1",
        "pending": [{"tool": "research_point", "summary": "a"}],
    }


def test_web_stream_reports_a_failed_run_as_an_error_event(monkeypatch, stub_seams):
    def boom(_model_id):
        raise RuntimeError("no model for you")

    monkeypatch.setattr(agent_module, "openrouter_model", boom)

    assert list(agent_module.stream_agent("what is innerdance?")) == [
        {"type": "error", "message": "no model for you"}
    ]
