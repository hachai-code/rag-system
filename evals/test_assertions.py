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

from app import AskRequest, _no_relevant_hits
from evals.metrics import recall_at_k, reciprocal_rank
from rag import RELEVANCE_THRESHOLD, _citations

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


def test_retrieval_metrics():
    """recall@k is a hit/miss flag; reciprocal rank is 1/rank of the first gold hit."""
    assert recall_at_k([3, 1, 2], {1}) == 1.0
    assert recall_at_k([3, 4, 5], {1}) == 0.0
    assert reciprocal_rank([3, 1, 2], {1}) == 0.5
    assert reciprocal_rank([3, 4, 5], {1}) == 0.0
