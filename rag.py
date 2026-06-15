"""Minimal RAG over the innerdance corpus: retrieve, then (later) ask Claude.

This is the query side of the pipeline that ingest -> chunk -> embed built.
Run `uv run rag.py` to see the top-k chunks for a sample question.
"""

import anthropic
import psycopg
import voyageai
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

load_dotenv()

DB_URL = "postgresql://postgres:postgres@localhost:5432/rag"
VOYAGE_MODEL = "voyage-4"
EMBED_DIM = 1024
TOP_K = 5

# Reciprocal Rank Fusion: pull this many candidates from each retriever, then
# fuse. RRF_K (60, the value from the original RRF paper) damps how much a chunk's
# exact rank matters, so one strong retriever can't dominate on rank alone.
FUSE_DEPTH = 60
RRF_K = 60

# Vector is the stronger retriever on this corpus, so keyword is down-weighted: it
# can still rescue queries vector misses, but can't displace a chunk vector ranked
# confidently. 0.5 won a weight sweep — recall@5 0.79 vs 0.74 for both pure vector
# and equal-weight fusion (see metrics_log.jsonl).
VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.5

# Reranking: pull this many candidates from hybrid search, then a cross-encoder
# re-scores each (query, chunk) pair and we keep the top TOP_K.
RERANK_MODEL = "rerank-2.5"
RERANK_DEPTH = 20

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
# No citation instructions here: the Citations feature handles attribution itself,
# returning the exact source quote for each claim, so we only ask for grounding.
SYSTEM_PROMPT = (
    "You answer questions about the innerdance corpus using only the provided "
    "documents. If the documents do not contain the answer, say you don't know."
)

_voyage = voyageai.Client()


def embed_query(question: str) -> list[float]:
    """Embed the question. input_type='query' is the search-side counterpart to
    the 'document' embeddings we stored — Voyage tunes the two differently."""
    result = _voyage.embed(
        [question], model=VOYAGE_MODEL, input_type="query", output_dimension=EMBED_DIM
    )
    return result.embeddings[0]


def search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks most similar to the question, nearest first."""
    embedding = embed_query(question)
    return conn.execute(
        """
        SELECT c.id, d.title, d.source, c.content,
               c.embedding <=> %(emb)s::vector AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %(emb)s::vector
        LIMIT %(k)s
        """,
        {"emb": embedding, "k": k},
    ).fetchall()


def keyword_search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks whose text best matches the question's keywords, using
    Postgres full-text search (ts_rank, a BM25-like lexical score).

    websearch_to_tsquery ANDs every term, which is too strict for a full question
    (one missing word and nothing matches), so we swap & for | to OR the terms:
    any overlap counts, and ts_rank then orders by how well/often they match. This
    is the lexical counterpart to vector `search` — it nails exact rare terms that
    embeddings blur together, but, being exact-lexeme, it can't see past typos."""
    return conn.execute(
        """
        WITH q AS (
            SELECT replace(
                websearch_to_tsquery('english', %(question)s)::text, '&', '|'
            )::tsquery AS tsq
        )
        SELECT c.id, d.title, d.source, c.content,
               ts_rank(c.content_tsv, q.tsq) AS rank
        FROM chunks c
        JOIN documents d ON d.id = c.document_id, q
        WHERE c.content_tsv @@ q.tsq
        ORDER BY rank DESC
        LIMIT %(k)s
        """,
        {"question": question, "k": k},
    ).fetchall()


def hybrid_search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[dict]:
    """Fuse vector and keyword rankings with weighted Reciprocal Rank Fusion (RRF).

    Each chunk scores sum(weight / (RRF_K + rank)) over the two rankings it appears
    in, so a chunk ranked high by either retriever rises, and one ranked high by
    both wins. RRF combines on *rank*, not score, which is why it blends cosine
    distance and ts_rank — two scales that aren't comparable — without normalizing.
    Keyword is down-weighted (see KEYWORD_WEIGHT) because it's the noisier signal."""
    rankings = [
        (VECTOR_WEIGHT, search(conn, question, FUSE_DEPTH)),
        (KEYWORD_WEIGHT, keyword_search(conn, question, FUSE_DEPTH)),
    ]
    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for weight, hits in rankings:
        for rank, hit in enumerate(hits, 1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + weight / (RRF_K + rank)
            chunks[hit["id"]] = hit
    top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [chunks[cid] for cid in top_ids]


def rerank_search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[dict]:
    """Three-stage retrieval: hybrid retrieve RERANK_DEPTH candidates, rerank, keep k.

    Vector and keyword search score the question and a chunk *independently*, so a
    chunk only has to land near the question to rank well. A cross-encoder reads the
    (question, chunk) pair together and judges how well the chunk actually answers
    the question — a sharper signal that rescues near-misses the first stage
    over-ranks. It's too slow to run over the whole corpus, so it only re-scores the
    RERANK_DEPTH candidates hybrid search already narrowed to."""
    candidates = hybrid_search(conn, question, RERANK_DEPTH)
    reranked = _voyage.rerank(
        question, [c["content"] for c in candidates], model=RERANK_MODEL, top_k=k
    )
    return [candidates[r.index] for r in reranked.results]


def context_documents(hits: list[dict]) -> list[dict]:
    """Turn each retrieved chunk into a citable document content block.

    Passing chunks as separate documents (rather than one stuffed prompt) is what
    lets the Citations feature attribute claims: each document's `document_index`
    in a returned citation maps straight back to hits[index]."""
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit["content"]},
            "title": hit["title"],
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def answer(question: str, hits: list[dict]) -> tuple[str, list[dict]]:
    """Ask Claude over the retrieved chunks and return (answer_text, citations).

    With citations enabled the response is a sequence of text blocks; a block that
    makes a claim carries a `.citations` list, each pointing at the exact source
    text. We flatten the blocks into the answer string and collect one record per
    citation, tying each cited claim back to the chunk it came from so the frontend
    can link it. `cited_text` is extracted by the API from the document, so it can't
    be a quote the source doesn't contain."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": context_documents(hits)
                + [{"type": "text", "text": f"Question: {question}"}],
            }
        ],
    )
    text = "".join(block.text for block in response.content)
    citations = [
        {
            "claim": block.text,
            "cited_text": c.cited_text,
            "chunk_id": hits[c.document_index]["id"],
            "title": hits[c.document_index]["title"],
            "source": hits[c.document_index]["source"],
        }
        for block in response.content
        for c in (block.citations or [])
    ]
    return text, citations


if __name__ == "__main__":
    question = "What is the relationship between epilepsy and spiritual experience?"
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = rerank_search(conn, question)

    text, citations = answer(question, hits)
    print(f"Q: {question}\n")
    print(text)

    by_id = {hit["id"]: hit["content"] for hit in hits}
    print(f"\nCitations ({len(citations)}):")
    for cite in citations:
        # Groundedness check: the API extracts cited_text from the chunk, so this
        # should always hold — if it ever fails, the citation is not real.
        grounded = cite["cited_text"] in by_id[cite["chunk_id"]]
        mark = "ok" if grounded else "HALLUCINATED"
        print(f"  [{mark}] {cite['title'][:40]} — \"{cite['cited_text'][:70]}\"")
