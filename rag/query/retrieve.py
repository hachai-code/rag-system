"""Query stage 1: retrieve the chunks most relevant to a question.

Three-stage retrieval — hybrid (vector + keyword, fused with RRF) then rerank. See
README "Retrieval design" for the weights, thresholds, and why each stage exists.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import psycopg
from langfuse import get_client

from ..clients import voyage_client
from ..config import CONFIG
from ..db import EMBED_DIM, Hit, connect
from .answer import complete

VOYAGE_MODEL = CONFIG.voyage_model
TOP_K = CONFIG.top_k
METHOD = CONFIG.method  # production default retriever (vector | hybrid | rerank)

# Runtime query-enhancement toggles; "" in config means off.
QUERY_ENHANCEMENT = CONFIG.query_enhancement or None
PARENT_DOCUMENT = CONFIG.parent_document
HYPE = CONFIG.hype  # match against index-time hypothetical questions (see hype_search)
# Cheap model for HyDE hypotheticals and multi-query paraphrases (the "flash" picker).
FLASH_MODEL = CONFIG.gen_models["flash"]
N_VARIANTS = 4  # multi-query paraphrases fused with the original question

# Distance beyond which the corpus is treated as not covering the question.
RELEVANCE_THRESHOLD = CONFIG.relevance_threshold

# Reciprocal Rank Fusion: candidates pulled per retriever, and the paper's k=60.
FUSE_DEPTH = CONFIG.fuse_depth
RRF_K = CONFIG.rrf_k

# Keyword is the noisier retriever, so it's down-weighted.
VECTOR_WEIGHT = CONFIG.vector_weight
KEYWORD_WEIGHT = CONFIG.keyword_weight

RERANK_MODEL = CONFIG.rerank_model
RERANK_DEPTH = CONFIG.rerank_depth

# Chunks of surrounding context to show on each side of a cited chunk (click-through).
SOURCE_WINDOW = CONFIG.source_window


@lru_cache(maxsize=256)
def embed_query(question: str) -> list[float]:
    """Embed the question. input_type='query' is the search-side counterpart to the
    stored 'document' embeddings — Voyage tunes the two differently."""
    result = voyage_client().embed(
        [question], model=VOYAGE_MODEL, input_type="query", output_dimension=EMBED_DIM
    )
    return result.embeddings[0]


def search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[Hit]:
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


def no_relevant_hits(hits: list[Hit], threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """True when retrieval found nothing close enough to answer from."""
    return not hits or hits[0]["distance"] > threshold


def covered(
    conn: psycopg.Connection, question: str, threshold: float = RELEVANCE_THRESHOLD
) -> tuple[bool, list[Hit]]:
    """Cheap coverage gate shared by the API and the eval runner: probe with k=1 and
    apply the relevance threshold. Also returns the probe hits so callers can log
    what the gate saw. Keeping prod and eval on this one function means the gate
    can't drift between them."""
    gate = search(conn, question, k=1)
    return not no_relevant_hits(gate, threshold), gate


def keyword_search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[Hit]:
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


def rrf(ranked_lists: list[tuple[float, list[Hit]]], k: int = RRF_K) -> list[Hit]:
    """Weighted Reciprocal Rank Fusion: merge several ranked hit lists into one.

    Each chunk scores sum(weight / (k + rank)) over the lists it appears in — fusing on
    rank (not score) is what lets it blend cosine distance, ts_rank, and multiple query
    phrasings. Returns all fused hits, best first; callers slice to their top_k."""
    scores: dict[int, float] = {}
    chunks: dict[int, Hit] = {}
    for weight, hits in ranked_lists:
        for rank, hit in enumerate(hits, 1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + weight / (k + rank)
            chunks[hit["id"]] = hit
    return [chunks[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


def hybrid_search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[Hit]:
    """Fuse vector and keyword rankings with weighted Reciprocal Rank Fusion.

    The two retrievers pull FUSE_DEPTH candidates each before rrf() blends them."""
    ranked_lists = [
        (VECTOR_WEIGHT, search(conn, question, FUSE_DEPTH)),
        (KEYWORD_WEIGHT, keyword_search(conn, question, FUSE_DEPTH)),
    ]
    return rrf(ranked_lists)[:k]


def _rerank(question: str, candidates: list[Hit], k: int) -> list[Hit]:
    """Cross-encoder rerank a candidate list against the real question, keep k.

    Reused by any first stage that wants rerank on top — hybrid (rerank_search) or the
    HyPE question-match path — so all of them score (question, chunk) the same way."""
    if not candidates:
        return []
    reranked = voyage_client().rerank(
        question, [c["content"] for c in candidates], model=RERANK_MODEL, top_k=k
    )
    return [candidates[r.index] for r in reranked.results]


def rerank_search(
    conn: psycopg.Connection, question: str, k: int = TOP_K, retrieve_query: str | None = None
) -> list[Hit]:
    """Hybrid-retrieve RERANK_DEPTH candidates, cross-encoder rerank, keep k.

    The reranker reads the (question, chunk) pair together — a sharper signal than the
    independent first-stage scores — but is too slow to run over the whole corpus.

    `retrieve_query` overrides the stage-1 query only (the HyDE hypothetical, which lands
    nearer the answer chunks in embedding space); the reranker still scores against the
    real `question`, where the cross-encoder is sharpest."""
    candidates = hybrid_search(conn, retrieve_query or question, RERANK_DEPTH)
    return _rerank(question, candidates, k)


# Retrieval stages share the (conn, question, k) signature, so a config can pick the
# funnel depth — vector-only, +keyword/RRF, or +rerank — without a code edit.
RETRIEVERS = {"vector": search, "hybrid": hybrid_search, "rerank": rerank_search}


def get_retriever(method: str = METHOD):
    """Map a retrieval.method config string to its retriever function."""
    return RETRIEVERS[method]


def _dedupe_to_parent(hits: list[Hit]) -> list[Hit]:
    """Collapse question-level matches to their distinct parent chunks, keeping each
    chunk's best (smallest) distance. Several hypothetical questions can point at the
    same chunk; we want one hit per chunk, ordered nearest first."""
    best: dict[int, Hit] = {}
    for hit in hits:
        current = best.get(hit["id"])
        if current is None or hit["distance"] < current["distance"]:
            best[hit["id"]] = hit
    return sorted(best.values(), key=lambda h: h["distance"])


def hype_search(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[Hit]:
    """HyPE: match the query against index-time hypothetical questions, then map each hit
    back to its parent chunk.

    The stored question vectors sit closer to real queries than the raw chunks do, so this
    closes the query/document phrasing gap. We over-pull FUSE_DEPTH question matches, then
    dedupe to distinct parent chunks — the served `content` is the raw chunk, not the
    hypothetical question."""
    embedding = embed_query(question)
    hits = conn.execute(
        """
        SELECT c.id, d.title, d.source, c.content,
               cq.embedding <=> %(emb)s::vector AS distance
        FROM chunk_questions cq
        JOIN chunks c ON c.id = cq.chunk_id
        JOIN documents d ON d.id = c.document_id
        ORDER BY cq.embedding <=> %(emb)s::vector
        LIMIT %(depth)s
        """,
        {"emb": embedding, "depth": FUSE_DEPTH},
    ).fetchall()
    return _dedupe_to_parent(hits)[:k]


HYDE_PROMPT = (
    "Write a short, direct passage that plausibly answers this question about the "
    "innerdance corpus, as if excerpted from source material. Do not hedge or note "
    "uncertainty — just write the hypothetical answer.\n\nQuestion: {question}"
)

MULTI_QUERY_PROMPT = (
    "Rewrite this question as {n} alternative search queries that ask the same thing "
    "in different words, to widen retrieval. One per line, no numbering or "
    "commentary.\n\nQuestion: {question}"
)


def hyde_query(question: str) -> str:
    """A hypothetical answer to retrieve on instead of the question (HyDE).

    A made-up answer lands nearer the real answer chunks in embedding space than the
    terse question does, so retrieving on it surfaces passages the bare question ranks
    too low."""
    return complete(HYDE_PROMPT.format(question=question), FLASH_MODEL)


@lru_cache(maxsize=256)
def multi_query(question: str) -> tuple[str, ...]:
    """Up to N_VARIANTS paraphrases of the question, for retrieval fusion.

    Different phrasings surface different chunks; fusing their rankings recovers a
    passage any single wording would miss. Cached so a repeated question in a run
    doesn't re-call the LLM."""
    text = complete(MULTI_QUERY_PROMPT.format(n=N_VARIANTS, question=question), FLASH_MODEL)
    variants = [line.strip() for line in text.splitlines() if line.strip()]
    return tuple(variants[:N_VARIANTS])


def _parent_range(chunk_index: int, window: int) -> tuple[int, int]:
    """The inclusive chunk_index span to pull around a matched child chunk."""
    return chunk_index - window, chunk_index + window


def expand_to_parent(
    conn: psycopg.Connection, hits: list[Hit], window: int = SOURCE_WINDOW
) -> list[Hit]:
    """Widen each matched child chunk to its neighbours in the same document.

    A 256-token child can match on a phrase whose answer lives in the surrounding
    passage — a transcript utterance whose reply is the next chunk. This pulls the
    ±window chunks around each hit (same neighbour navigation as source_passage), so the
    generator sees the parent section. Duplicates are dropped; added neighbours carry no
    retrieval distance."""
    hit_distance = {h["id"]: h.get("distance") for h in hits}
    expanded: dict[int, Hit] = {}
    for hit in hits:
        target = conn.execute(
            "SELECT document_id, chunk_index FROM chunks WHERE id = %s", (hit["id"],)
        ).fetchone()
        lo, hi = _parent_range(target["chunk_index"], window)
        rows = conn.execute(
            """SELECT c.id, d.title, d.source, c.content
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.document_id = %s AND c.chunk_index BETWEEN %s AND %s
               ORDER BY c.chunk_index""",
            (target["document_id"], lo, hi),
        ).fetchall()
        for row in rows:
            expanded.setdefault(row["id"], {**row, "distance": hit_distance.get(row["id"])})
    return list(expanded.values())


def _retrieve_fresh(retriever, question: str, k: int) -> list[Hit]:
    """Run one retrieval on its own connection — multi-query fans out across threads,
    and psycopg connections are not shared between them."""
    with connect() as conn:
        return retriever(conn, question, k)


def _multi_query_search(question: str, method: str, k: int) -> list[Hit]:
    """Retrieve for the question and each paraphrase concurrently, then RRF-fuse."""
    queries = (question,) + multi_query(question)
    retriever = get_retriever(method)
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        ranked_lists = [
            (1.0, hits)
            for hits in pool.map(lambda q: _retrieve_fresh(retriever, q, FUSE_DEPTH), queries)
        ]
    return rrf(ranked_lists)[:k]


def retrieve(
    conn: psycopg.Connection,
    question: str,
    k: int = TOP_K,
    method: str = METHOD,
    query_enhancement: str | None = QUERY_ENHANCEMENT,
    parent_document: bool = PARENT_DOCUMENT,
    hype: bool = HYPE,
) -> list[Hit]:
    """The retrieval entry point: run `method`, with optional query enhancement and
    parent-document expansion. app.py and the eval runner call this so a config, not a
    code edit, picks the retrieval strategy."""
    # One span for the whole retrieval stage, so traces show retrieval vs generation
    # time. get_client() is a no-op without LANGFUSE_* keys (scripts, evals).
    with get_client().start_as_current_observation(
        as_type="span",
        name="retrieval",
        input={
            "question": question,
            "method": method,
            "k": k,
            "query_enhancement": query_enhancement,
            "parent_document": parent_document,
            "hype": hype,
        },
    ) as span:
        hits = _retrieve(conn, question, k, method, query_enhancement, parent_document, hype)
        span.update(output={"n_hits": len(hits), "chunk_ids": [h["id"] for h in hits]})
        return hits


def _retrieve(
    conn: psycopg.Connection,
    question: str,
    k: int,
    method: str,
    query_enhancement: str | None,
    parent_document: bool,
    hype: bool,
) -> list[Hit]:
    if hype:
        # HyPE supplies the first stage — the query matched against hypothetical-question
        # vectors, mapped back to parent chunks. With rerank, the cross-encoder still
        # scores the real question for precision (same split as HyDE + rerank); vector/
        # hybrid just take HyPE's ranking as-is. `hype` is an independent toggle, so an
        # A/B with hype off falls through to the normal `method` path unchanged.
        if method == "rerank":
            candidates = hype_search(conn, question, RERANK_DEPTH)
            hits = _rerank(question, candidates, k)
        else:
            hits = hype_search(conn, question, k)
    elif query_enhancement == "hyde":
        hypothetical = hyde_query(question)
        if method == "rerank":
            # HyDE lifts stage-1 recall; rerank on the real question for precision.
            hits = rerank_search(conn, question, k, retrieve_query=hypothetical)
        else:
            hits = get_retriever(method)(conn, hypothetical, k)
    elif query_enhancement == "multi_query":
        hits = _multi_query_search(question, method, k)
    else:
        hits = get_retriever(method)(conn, question, k)
    if parent_document:
        hits = expand_to_parent(conn, hits)
    return hits


def _overlap(prefix_lines: list[str], next_lines: list[str]) -> int:
    """How many trailing lines of prefix_lines equal the leading lines of next_lines.

    Adjacent chunks share a CHUNK_OVERLAP-sized run of whole lines (see chunk.py)."""
    for k in range(min(len(prefix_lines), len(next_lines)), 0, -1):
        if prefix_lines[-k:] == next_lines[:k]:
            return k
    return 0


def _stitch(rows: list[Hit], target_index: int) -> tuple[str, str, str]:
    """Join consecutive chunks into (before, chunk, after), dropping overlap.

    The target chunk is kept whole so a citation's quote stays findable inside it."""
    before: list[str] = []
    chunk: list[str] = []
    after: list[str] = []
    for row in rows:
        lines = row["content"].split("\n")
        if row["chunk_index"] < target_index:
            before.extend(lines[_overlap(before, lines) :])
        elif row["chunk_index"] == target_index:
            chunk = lines
            del before[len(before) - _overlap(before, lines) :]  # trim before's tail
        else:
            after.extend(lines[_overlap(chunk + after, lines) :])
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
    heading = next((r["heading"] for r in rows if r["chunk_index"] == target["chunk_index"]), None)
    return {
        "title": doc["title"],
        "section": heading or doc["section"],
        "chunk_index": target["chunk_index"],
        "n_chunks": n_chunks,
        "before": before,
        "chunk": chunk,
        "after": after,
    }
