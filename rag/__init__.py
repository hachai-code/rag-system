"""innerdance RAG. Pipeline order and design notes live in the README.

Re-exports the public surface so callers can `from rag import search, answer, ...`
regardless of which submodule a symbol lives in.
"""

from dotenv import load_dotenv

load_dotenv()

from rag.db import DB_URL
from rag.query.answer import (
    ANSWER_FORMAT,
    GEN_MODEL,
    GEN_PROVIDER,
    OPENROUTER_BASE_URL,
    STRUCTURED_MAX_TOKENS,
    SYSTEM_PROMPT,
    Claim,
    GroundedAnswer,
    _chunk_citations,
    _citations,
    answer,
    answer_stream,
)
from rag.query.retrieve import (
    RELEVANCE_THRESHOLD,
    TOP_K,
    expand_to_parent,
    get_retriever,
    hybrid_search,
    hype_search,
    keyword_search,
    rerank_search,
    retrieve,
    rrf,
    search,
    source_passage,
)

__all__ = [
    "DB_URL",
    "TOP_K",
    "RELEVANCE_THRESHOLD",
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
    "ANSWER_FORMAT",
    "GEN_MODEL",
    "GEN_PROVIDER",
    "OPENROUTER_BASE_URL",
    "STRUCTURED_MAX_TOKENS",
    "SYSTEM_PROMPT",
    "Claim",
    "GroundedAnswer",
    "_chunk_citations",
    "answer",
    "answer_stream",
    "_citations",
]
