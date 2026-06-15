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
SYSTEM_PROMPT = (
    "You answer questions about the innerdance corpus using only the provided "
    "context passages. Cite the passages you use by number, like [1]. If the "
    "context does not contain the answer, say you don't know."
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


def format_context(hits: list[dict]) -> str:
    """Number each retrieved chunk so Claude can cite it as [1], [2], ..."""
    blocks = [f"[{i}] {hit['title']}\n{hit['content']}" for i, hit in enumerate(hits, 1)]
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict]) -> str:
    """Stuff the retrieved chunks into the prompt and return Claude's answer."""
    context = format_context(hits)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
    )
    return response.content[0].text


if __name__ == "__main__":
    question = "What is the relationship between epilepsy and spiritual experience?"
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = search(conn, question)

    print(f"Q: {question}\n")
    print(answer(question, hits))
    print("\nSources:")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit['title'][:60]}")
