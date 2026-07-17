"""The `/ask` gate: coverage check, retrieval, then a grounded answer.

One source of truth for the "search -> relevance threshold -> answer" sequence that
app.py's /ask serves and the eval runners judge against. Keeping prod and eval on this
one function is what stops the gate drifting between them (the eval copies used to run
a plain vector `search()` while /ask ran the full `retrieve()` funnel).
"""

from dataclasses import dataclass

import psycopg

from ..db import Hit
from .answer import ANSWER_FORMAT, GEN_MODEL, Citation, answer
from .retrieve import (
    HYPE,
    METHOD,
    PARENT_DOCUMENT,
    QUERY_ENHANCEMENT,
    TOP_K,
    covered,
    retrieve,
)

NO_ANSWER = "I don't have information on that in the innerdance corpus."


@dataclass
class GateResult:
    """The outcome of the gate. `hits` are the coverage-probe hits when `gated`, else the
    retrieved chunks that produced the answer; callers adapt this to their own response."""

    gated: bool
    hits: list[Hit]
    answer: str
    citations: list[Citation]


def gate_retrieve(
    conn: psycopg.Connection,
    question: str,
    *,
    k: int = TOP_K,
    method: str = METHOD,
    query_enhancement: str | None = QUERY_ENHANCEMENT,
    parent_document: bool = PARENT_DOCUMENT,
    hype: bool = HYPE,
) -> tuple[bool, list[Hit]]:
    """The coverage-check + retrieval prefix, shared by ask_gate and /ask/stream.

    Returns (gated, hits): when the corpus doesn't cover the question, `gated` is True and
    `hits` are the cheap probe hits (for logging); otherwise `hits` are the retrieved chunks."""
    ok, probe = covered(conn, question)
    if not ok:
        return True, probe
    hits = retrieve(
        conn,
        question,
        k=k,
        method=method,
        query_enhancement=query_enhancement,
        parent_document=parent_document,
        hype=hype,
    )
    return False, hits


def ask_gate(
    conn: psycopg.Connection,
    question: str,
    *,
    k: int = TOP_K,
    method: str = METHOD,
    query_enhancement: str | None = QUERY_ENHANCEMENT,
    parent_document: bool = PARENT_DOCUMENT,
    hype: bool = HYPE,
    model: str = GEN_MODEL,
    fmt: str = ANSWER_FORMAT,
) -> GateResult:
    """Answer `question` from the corpus, or refuse when nothing relevant was retrieved."""
    gated, hits = gate_retrieve(
        conn,
        question,
        k=k,
        method=method,
        query_enhancement=query_enhancement,
        parent_document=parent_document,
        hype=hype,
    )
    if gated:
        return GateResult(gated=True, hits=hits, answer=NO_ANSWER, citations=[])
    text, citations = answer(question, hits, model=model, fmt=fmt)
    return GateResult(gated=False, hits=hits, answer=text, citations=citations)
