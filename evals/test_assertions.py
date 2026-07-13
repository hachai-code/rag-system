"""Level-1 assertions: deterministic checks that need no DB, model, or network.

These are the cheap guardrails from DESIGN.md — fast enough to run on every commit
in CI. Anything that needs the live DB or an API call (retrieval determinism, an
end-to-end answer) is an integration check, not a unit assertion, and deliberately
stays out of this file so CI never depends on a database or API key.

Run: uv run pytest
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag.app import AskRequest, _no_relevant_hits
from rag.query.deepagent import _step_label, format_hits_for_deepagent
from rag.query.retrieve import _dedupe_to_parent, _parent_range, _rerank
from evals.search.metrics import recall_at_k, reciprocal_rank
from rag import (
    RELEVANCE_THRESHOLD,
    Claim,
    GroundedAnswer,
    _chunk_citations,
    _citations,
    get_retriever,
    hybrid_search,
    rerank_search,
    rrf,
    search,
)

# Two chunks standing in for retrieved hits. `_citations` indexes into this list by
# the citation's document_index, exactly as the live code does.
HITS = [
    {"id": 11, "title": "Day 1", "source": "day1.rtf",
     "content": "dopamine reroutes to the old brain"},
    {"id": 22, "title": "Day 2", "source": "day2.rtf",
     "content": "stillness is where innerdance begins"},
]


def _block(text, citations=()):
    """Fake an Anthropic response block: a `.text` and a `.citations` list, the only
    two attributes `_citations` reads."""
    return SimpleNamespace(text=text, citations=list(citations))


def _cite(cited_text, document_index):
    return SimpleNamespace(cited_text=cited_text, document_index=document_index)


def test_cited_quote_is_grounded_in_its_chunk():
    """The quote on a citation must be a substring of the chunk it points at — if it
    isn't, the citation is fabricated."""
    content = [_block("It reroutes.", [_cite("dopamine reroutes to the old brain", 0)])]
    [cite] = _citations(content, HITS)
    assert cite["cited_text"] in HITS[0]["content"]


def test_citation_maps_back_to_its_hit():
    content = [_block("x", [_cite("stillness is where innerdance begins", 1)])]
    [cite] = _citations(content, HITS)
    assert (cite["chunk_id"], cite["title"], cite["source"]) == (22, "Day 2", "day2.rtf")


def test_blocks_without_citations_produce_no_records():
    assert _citations([_block("Unsupported claim.", [])], HITS) == []


# The OpenAI-compatible path cites whole retrieved chunks rather than model-written
# quotes: the model returns claims tagged with chunk indices, and _chunk_citations maps
# each index back to its hit. cited_text is the chunk itself (grounded by construction),
# and an out-of-range index — a chunk the model named that wasn't retrieved — is dropped.
def _grounded(claims: list[dict]) -> GroundedAnswer:
    return GroundedAnswer.model_validate({"claims": claims})


def test_chunk_citation_maps_to_its_hit():
    grounded = _grounded([{"statement": "It reroutes.", "chunk_indices": [0]}])
    text, [cite] = _chunk_citations(grounded, HITS)
    assert text == "It reroutes."
    assert cite["cited_text"] == HITS[0]["content"]
    assert (cite["chunk_id"], cite["title"], cite["source"]) == (11, "Day 1", "day1.rtf")


def test_out_of_range_chunk_index_is_dropped():
    grounded = _grounded([{"statement": "Cites a chunk that wasn't retrieved.",
                           "chunk_indices": [99]}])
    text, citations = _chunk_citations(grounded, HITS)
    assert text == "Cites a chunk that wasn't retrieved."
    assert citations == []


def test_no_answer_gate():
    """The gate refuses when nothing was retrieved or the nearest hit is past the
    relevance threshold, and answers otherwise."""
    assert _no_relevant_hits([]) is True
    assert _no_relevant_hits([{"distance": RELEVANCE_THRESHOLD + 0.01}]) is True
    assert _no_relevant_hits([{"distance": RELEVANCE_THRESHOLD - 0.01}]) is False


def test_question_length_is_bounded():
    AskRequest(question="What is innerdance?")  # a normal question is accepted
    with pytest.raises(ValidationError):
        AskRequest(question="")
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 1001)


def test_get_retriever_maps_method_to_function():
    """The retrieval.method config string picks the funnel depth; each maps to its
    retriever so configs can A/B vector vs hybrid vs rerank without a code edit."""
    assert get_retriever("vector") is search
    assert get_retriever("hybrid") is hybrid_search
    assert get_retriever("rerank") is rerank_search


def test_retrieval_metrics():
    """recall@k is a hit/miss flag; reciprocal rank is 1/rank of the first gold hit."""
    assert recall_at_k([3, 1, 2], {1}) == 1.0
    assert recall_at_k([3, 4, 5], {1}) == 0.0
    assert reciprocal_rank([3, 1, 2], {1}) == 0.5
    assert reciprocal_rank([3, 4, 5], {1}) == 0.0


def test_rrf_fuses_ranked_lists_into_one_order():
    """Weighted RRF scores each chunk sum(weight / (k + rank)) over the lists it appears
    in, so agreement across lists and a heavier weight both lift a chunk. The shared
    helper drives hybrid_search and multi-query fusion alike."""
    vector = [{"id": 1}, {"id": 2}, {"id": 3}]  # ranks 1, 2, 3
    keyword = [{"id": 3}, {"id": 4}]            # ranks 1, 2
    fused = [h["id"] for h in rrf([(1.0, vector), (0.5, keyword)], k=60)]
    # 3 wins on cross-list agreement; 1 > 2 by rank; 4 last on the down-weighted list.
    assert fused == [3, 1, 2, 4]


def test_parent_range_spans_neighbours():
    """expand_to_parent pulls the ±window chunks around a matched child by chunk_index —
    the same neighbour window source_passage uses for click-through."""
    assert _parent_range(10, window=3) == (7, 13)
    assert _parent_range(0, window=2) == (-2, 2)  # SQL BETWEEN simply matches no chunk < 0


def test_hype_dedupes_to_parent_keeping_min_distance():
    """HyPE matches the query against hypothetical questions, so several matches can point
    at the same parent chunk; dedupe keeps one hit per chunk at its best distance, nearest
    first. The served `content` is the raw parent chunk — the question is only what was
    matched, never what's stored on the chunk or shown to the generator."""
    matches = [
        {"id": 11, "title": "Day 1", "source": "day1.rtf",
         "content": "dopamine reroutes to the old brain", "distance": 0.40},
        {"id": 22, "title": "Day 2", "source": "day2.rtf",
         "content": "stillness is where innerdance begins", "distance": 0.30},
        {"id": 11, "title": "Day 1", "source": "day1.rtf",
         "content": "dopamine reroutes to the old brain", "distance": 0.20},
    ]
    deduped = _dedupe_to_parent(matches)
    assert [(h["id"], h["distance"]) for h in deduped] == [(11, 0.20), (22, 0.30)]
    assert deduped[0]["content"] == "dopamine reroutes to the old brain"


def test_rerank_empty_candidates_returns_empty():
    """HyPE + rerank feeds the reranker whatever HyPE found; if chunk_questions is empty
    (HyPE not populated), the candidate list is empty and rerank must no-op rather than
    call the cross-encoder on nothing."""
    assert _rerank("any question", [], k=5) == []


def test_format_hits_for_deepagent_is_numbered_and_citable():
    """The retrieve_corpus tool renders hits as numbered, cite-able passages: every
    hit appears with a 1-based index the agent can cite by, plus its title and text."""
    text = format_hits_for_deepagent(HITS)
    for i, hit in enumerate(HITS, 1):
        assert f"[{i}]" in text
        assert hit["title"] in text
        assert hit["content"] in text


def test_format_hits_registry_keeps_stable_numbers():
    """With a shared registry, each chunk keeps a stable [n] across retrievals: a chunk
    seen earlier reuses its number instead of renumbering to [1], and the registry
    records the chunk_id so the frontend can open the passage."""
    registry: dict = {}
    first = format_hits_for_deepagent(HITS, registry)
    assert "[1]" in first and "[2]" in first
    again = format_hits_for_deepagent([HITS[1]], registry)  # re-retrieve the 2nd hit alone
    assert "[2]" in again and "[1]" not in again
    assert registry[HITS[1]["id"]]["chunk_id"] == HITS[1]["id"]


def test_over_research_budget_caps_web_calls():
    """The web tools charge one call each against a per-run budget keyed by thread_id;
    once the budget is spent the tool is told to stop. No thread_id means no cap."""
    from rag.query.deepagent import RESEARCH_BUDGET, _budgets, _over_research_budget, _web_calls

    cfg = {"configurable": {"thread_id": "t-budget"}}
    _web_calls["t-budget"] = RESEARCH_BUDGET - 1
    assert _over_research_budget(cfg) is False  # this call spends exactly the budget
    assert _over_research_budget(cfg) is True  # one past the budget → stop
    assert _over_research_budget({"configurable": {}}) is False  # untracked run, no cap

    # A per-run budget of 0 turns the cap off — the tool never gets told to stop.
    _budgets["t-budget"] = 0
    assert _over_research_budget(cfg) is False
    _web_calls.pop("t-budget", None)
    _budgets.pop("t-budget", None)


def test_step_label_summarizes_each_tool_call():
    """The live trace labels each tool call by its intent, folding in the salient arg
    (query/url/description). Unknown tools fall back to their raw name."""
    assert "stillness" in _step_label("retrieve_corpus", {"query": "stillness"})
    assert "example.com" in _step_label("fetch_page", {"url": "https://example.com"})
    assert _step_label("write_todos", {}) == "Planning the research"
    assert _step_label("mystery_tool", {}) == "mystery_tool"


def test_answer_stream_prose_streams_tokens_and_honors_system(monkeypatch):
    """The openai-compat prose path token-streams (one text event per delta, not one
    blob) and threads a `system` override through to the completion — so a prompt-variant
    config reaches /ask/stream, not just /ask. The completion client is faked, so no
    network: this only exercises the branching contract."""
    import rag.query.answer as answer_mod

    captured = {}

    def fake_create(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        captured["stream"] = kwargs.get("stream")
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
                for c in ["still", "ness ", "begins"]]

    fake_client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(answer_mod, "OpenAI", lambda **_: fake_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    events = list(answer_mod.answer_stream(
        "q?", HITS, provider="openai-compat", fmt="prose", system="OVERRIDE"))
    texts = [e["text"] for e in events if e["type"] == "text"]
    cites = [e for e in events if e["type"] == "citation"]
    assert texts == ["still", "ness ", "begins"]  # streamed per-token, not one event
    assert captured["system"] == "OVERRIDE" and captured["stream"] is True
    assert len(cites) == len(HITS)  # retrieved chunks emitted as proof


def test_unknown_tool_error_redirects_to_real_tools():
    from rag.query.web_search_agent import _execute_tool

    result = _execute_tool("write_todos", '{"todos": []}')
    assert result.startswith("Tool error")
    assert "search_web" in result and "fetch_page" in result  # break tool fixation


def test_todo_loop_breaker_blocks_noop_write_todos():
    from rag.query.deepagent import _TodoLoopBreaker

    todos = [{"content": "research", "status": "completed"}]
    request = SimpleNamespace(
        tool_call={"name": "write_todos", "args": {"todos": todos}, "id": "tc1"},
        state={"todos": todos},
    )

    def handler(_):
        raise AssertionError("no-op write_todos must not reach the tool")

    result = _TodoLoopBreaker().wrap_tool_call(request, handler)
    assert "final answer" in result.content


def test_todo_loop_breaker_passes_through_real_updates():
    from rag.query.deepagent import _TodoLoopBreaker

    request = SimpleNamespace(
        tool_call={"name": "write_todos",
                   "args": {"todos": [{"content": "new", "status": "pending"}]},
                   "id": "tc1"},
        state={"todos": []},
    )
    sentinel = object()
    assert _TodoLoopBreaker().wrap_tool_call(request, lambda _: sentinel) is sentinel


def test_final_answer_falls_back_to_last_ai_text():
    from langchain_core.messages import AIMessage, HumanMessage

    from rag.query.deepagent import DeepAnswer, _final_answer

    assert _final_answer({"structured_response": DeepAnswer(answer="typed")}) == "typed"
    assert _final_answer({"messages": [
        HumanMessage(content="q"), AIMessage(content="prose answer"),
    ]}) == "prose answer"
