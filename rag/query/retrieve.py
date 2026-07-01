"""Query stage 1: retrieve the chunks most relevant to a question.

Three-stage retrieval — hybrid (vector + keyword, fused with RRF) then rerank. See
README "Retrieval design" for the weights, thresholds, and why each stage exists.
"""

from functools import lru_cache

import psycopg
import voyageai

from ..config import CONFIG

VOYAGE_MODEL = CONFIG.voyage_model
EMBED_DIM = 1024
TOP_K = CONFIG.top_k

# Distance beyond which the corpus is treated as not covering the question (README).
RELEVANCE_THRESHOLD = CONFIG.relevance_threshold

# Reciprocal Rank Fusion: candidates pulled per retriever, and the paper's k=60.
FUSE_DEPTH = CONFIG.fuse_depth
RRF_K = CONFIG.rrf_k

# Keyword is the noisier retriever, so it's down-weighted (README "Retrieval design").
VECTOR_WEIGHT = CONFIG.vector_weight
KEYWORD_WEIGHT = CONFIG.keyword_weight

RERANK_MODEL = CONFIG.rerank_model
RERANK_DEPTH = CONFIG.rerank_depth

# Chunks of surrounding context to show on each side of a cited chunk (click-through).
SOURCE_WINDOW = CONFIG.source_window

_voyage = voyageai.Client(max_retries=2)


@lru_cache(maxsize=256)
def embed_query(question: str) -> list[float]:
    """Embed the question. input_type='query' is the search-side counterpart to the
    stored 'document' embeddings — Voyage tunes the two differently."""
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
    """Return the k chunks whose text best matches the question's keywords, via
    Postgres full-text search.

    websearch_to_tsquery ANDs every term (too strict for a full question), so we swap
    & for | to OR them — any overlap counts and ts_rank orders by match quality."""
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
    """Fuse vector and keyword rankings with weighted Reciprocal Rank Fusion.

    Each chunk scores sum(weight / (RRF_K + rank)) over the rankings it appears in.
    Fusing on rank (not score) is what lets it blend cosine distance and ts_rank."""
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
    """Hybrid-retrieve RERANK_DEPTH candidates, cross-encoder rerank, keep k.

    The reranker reads the (question, chunk) pair together — a sharper signal than the
    independent first-stage scores — but is too slow to run over the whole corpus."""
    candidates = hybrid_search(conn, question, RERANK_DEPTH)
    reranked = _voyage.rerank(
        question, [c["content"] for c in candidates], model=RERANK_MODEL, top_k=k
    )
    return [candidates[r.index] for r in reranked.results]


def _overlap(prefix_lines: list[str], next_lines: list[str]) -> int:
    """How many trailing lines of prefix_lines equal the leading lines of next_lines.

    Adjacent chunks share a CHUNK_OVERLAP-sized run of whole lines (see chunk.py)."""
    for k in range(min(len(prefix_lines), len(next_lines)), 0, -1):
        if prefix_lines[-k:] == next_lines[:k]:
            return k
    return 0


def _stitch(rows: list[dict], target_index: int) -> tuple[str, str, str]:
    """Join consecutive chunks into (before, chunk, after), dropping overlap.

    The target chunk is kept whole so a citation's quote stays findable inside it."""
    before: list[str] = []
    chunk: list[str] = []
    after: list[str] = []
    for row in rows:
        lines = row["content"].split("\n")
        if row["chunk_index"] < target_index:
            before.extend(lines[_overlap(before, lines):])
        elif row["chunk_index"] == target_index:
            chunk = lines
            del before[len(before) - _overlap(before, lines):]  # trim before's tail
        else:
            after.extend(lines[_overlap(chunk + after, lines):])
    return "\n".join(before), "\n".join(chunk), "\n".join(after)


def source_passage(conn: psycopg.Connection, chunk_id: int, window: int = SOURCE_WINDOW) -> dict:
    """Reconstruct where a cited chunk sits in its document, for click-through.

    Returns the chunk stitched back with `window` chunks of context on each side, plus
    the document title, the chunk's section (nearest heading, or the document's), and
    its position."""
    target = conn.execute(
        "SELECT document_id, chunk_index FROM chunks WHERE id = %s", (chunk_id,)
    ).fetchone()
    doc = conn.execute(
        "SELECT title, section FROM documents WHERE id = %s", (target["document_id"],)
    ).fetchone()
    n_chunks = conn.execute(
        "SELECT count(*) AS n FROM chunks WHERE document_id = %s", (target["document_id"],)
    ).fetchone()["n"]
    rows = conn.execute(
        """SELECT chunk_index, content, metadata->>'heading' AS heading
           FROM chunks
           WHERE document_id = %s AND chunk_index BETWEEN %s AND %s
           ORDER BY chunk_index""",
        (target["document_id"], target["chunk_index"] - window, target["chunk_index"] + window),
    ).fetchall()

    before, chunk, after = _stitch(rows, target["chunk_index"])
    heading = next(
        (r["heading"] for r in rows if r["chunk_index"] == target["chunk_index"]), None
    )
    return {
        "title": doc["title"],
        "section": heading or doc["section"],
        "chunk_index": target["chunk_index"],
        "n_chunks": n_chunks,
        "before": before,
        "chunk": chunk,
        "after": after,
    }
