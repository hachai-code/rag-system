"""innerdance RAG. Pipeline order and design notes live in the README.

Re-exports the public call surface so callers can `from rag import search, answer, ...`
regardless of which submodule a symbol lives in. Tuning constants and private helpers
are imported from their defining module.
"""

from dotenv import load_dotenv

load_dotenv()

from rag.query.answer import (
    Claim,
    GroundedAnswer,
    answer,
    answer_stream,
)
from rag.query.retrieve import (
    expand_to_parent,
    get_retriever,
    hybrid_search,
    hype_search,
    keyword_search,
    rerank_search,
    retrieve,
    rrf,
    search,
)
from rag.query.sources import source_passage

__all__ = [
    "search",
    "keyword_search",
    "hybrid_search",
    "rerank_search",
    "hype_search",
    "get_retriever",
    "retrieve",
    "rrf",
    "expand_to_parent",
    "source_passage",
    "Claim",
    "GroundedAnswer",
    "answer",
    "answer_stream",
]
