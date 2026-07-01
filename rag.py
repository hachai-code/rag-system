"""Minimal RAG over the innerdance corpus: retrieve, then (later) ask Claude.

This is the query side of the pipeline that ingest -> chunk -> embed built.
Run `uv run rag.py` to see the top-k chunks for a sample question.
"""

import os
from functools import lru_cache

import anthropic
import instructor
from instructor.core import IncompleteOutputException
import psycopg
import voyageai
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel

load_dotenv()

# Local docker default; set DATABASE_URL to point the app/pipeline at a remote DB.
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
VOYAGE_MODEL = "voyage-4"
EMBED_DIM = 1024
TOP_K = 5

# "No relevant results" gate: if even the nearest chunk is farther than this cosine
# distance, the corpus almost certainly doesn't cover the question, so the endpoint
# refuses ("I don't have information on that") instead of asking Claude — this both
# avoids hallucination and skips the generation cost on off-topic queries. Grounded
# in observed top-1 distances: on-topic ~0.40, a recoverable typo'd query reached
# 0.63, a genuine no-answer was 0.94 (see evals/failure-analysis.md). 0.7 sits above
# the recoverable band and below clear off-topic; retune against evals/eval_set.jsonl
# if the corpus or embedding model changes.
RELEVANCE_THRESHOLD = 0.7

# Reciprocal Rank Fusion: pull this many candidates from each retriever, then
# fuse. RRF_K (60, the value from the original RRF paper) damps how much a chunk's
# exact rank matters, so one strong retriever can't dominate on rank alone.
FUSE_DEPTH = 60
RRF_K = 60

# Vector is the stronger retriever on this corpus, so keyword is down-weighted: it
# can still rescue queries vector misses, but can't displace a chunk vector ranked
# confidently. 0.5 won a weight sweep — recall@5 0.79 vs 0.74 for both pure vector
# and equal-weight fusion (see evals/metrics_log.jsonl).
VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.5

# Reranking: pull this many candidates from hybrid search, then a cross-encoder
# re-scores each (query, chunk) pair and we keep the top TOP_K.
RERANK_MODEL = "rerank-2.5"
RERANK_DEPTH = 20

# Click-through to source: how many chunks of surrounding context to show on each
# side of a cited chunk when reconstructing its place in the document.
SOURCE_WINDOW = 3

CLAUDE_MODEL = "claude-sonnet-4-6"
# Provider seam: which adapter answer() dispatches to and which model it runs. Both
# default to the Anthropic baseline, so production is unchanged until config flips
# them (Phase 2/3). "anthropic" uses the native Citations API; "openai-compat" reaches
# OpenRouter through instructor and rebuilds citations from structured output.
GEN_PROVIDER = "anthropic"
GEN_MODEL = CLAUDE_MODEL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Per-query cost is bounded on every axis: the question length is capped at the API
# boundary (see app.py), retrieval sends a fixed TOP_K chunks as context, and
# MAX_TOKENS caps the generated output. So the worst-case spend per /ask is a known
# ceiling, not open-ended — and RELEVANCE_THRESHOLD skips this call entirely when
# nothing relevant was retrieved.
MAX_TOKENS = 1024
# The OpenAI-compatible path returns the same answer wrapped as JSON claims (statements
# plus their supporting chunk indices), which runs longer than the raw prose MAX_TOKENS
# the Anthropic path streams — a full answer serialized this way reaches ~2.5k tokens.
# Both models terminate cleanly at chunk granularity, so this is a real ceiling, not a
# guard against runaway generation.
STRUCTURED_MAX_TOKENS = 4096
# No citation instructions here: the Citations feature handles attribution itself,
# returning the exact source quote for each claim, so we only ask for grounding.
SYSTEM_PROMPT = (
    "You answer questions about the innerdance corpus using only the provided "
    "documents. If the documents do not contain the answer, say you don't know."
)

# Retry transient network errors (the Voyage client defaults to none, so a dropped
# connection during embedding otherwise fails hard). The Anthropic client already
# retries on its own; this gives embedding the same resilience on the /ask path and
# in the eval scripts.
_voyage = voyageai.Client(max_retries=2)


# Cache repeated queries: embeddings are deterministic for a given input, so an
# identical question never needs a second Voyage call — saves the embedding cost and
# its latency on repeats. The cached list is never mutated (it only feeds the SQL).
@lru_cache(maxsize=256)
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


def _messages(question: str, hits: list[dict]) -> list[dict]:
    """The user turn: the retrieved chunks as citable documents, then the question."""
    return [
        {
            "role": "user",
            "content": context_documents(hits)
            + [{"type": "text", "text": f"Question: {question}"}],
        }
    ]


def _openai_user_content(question: str, hits: list[dict]) -> str:
    """The OpenAI-compatible user turn: chunks numbered so the model can cite them by
    index, then the question. The numbering matches `hits` order, so a returned
    `chunk_index` maps straight back to hits[index] — the OpenAI-compat analogue of
    Anthropic's `document_index`."""
    numbered = "\n\n".join(
        f"[{i}] {hit['content']}" for i, hit in enumerate(hits)
    )
    return f"{numbered}\n\nQuestion: {question}"


def _citations(content: list, hits: list[dict]) -> list[dict]:
    """Collect one record per citation, tying each cited claim back to its chunk.

    With citations enabled the response is a sequence of text blocks; a block that
    makes a claim carries a `.citations` list pointing at the exact source text.
    `cited_text` is extracted by the API from the document, so it can't be a quote
    the source doesn't contain."""
    return [
        {
            "claim": block.text,
            "cited_text": c.cited_text,
            "chunk_id": hits[c.document_index]["id"],
            "title": hits[c.document_index]["title"],
            "source": hits[c.document_index]["source"],
        }
        for block in content
        for c in (block.citations or [])
    ]


# --- OpenAI-compatible citation rebuild ---------------------------------------
# Off-Anthropic there is no Citations API. Rather than have the model re-transcribe
# verbatim quotes (token-heavy, and weaker models run on unboundedly producing the
# JSON), we ask only which retrieved chunks support each claim — a list of indices,
# not text. The citation's `cited_text` is then the chunk itself, so it is grounded
# by construction and the model can't fabricate a quote. The trade-off vs the
# Anthropic path is granularity: a whole chunk, not the exact supporting sentence.


class Claim(BaseModel):
    """One sentence of the answer plus the indices of the chunks that support it.

    Indices are the [i] labels the chunks carry in the prompt (see
    _openai_user_content); they map straight back to hits[i]."""

    statement: str
    chunk_indices: list[int]


class GroundedAnswer(BaseModel):
    claims: list[Claim]


def _chunk_citations(grounded: GroundedAnswer, hits: list[dict]) -> tuple[str, list[dict]]:
    """Assemble (answer_text, citations) from claims tagged with chunk indices.

    The prose is the claims' statements joined in order; each supporting chunk becomes
    one citation record whose `cited_text` is the chunk's own content — the same dict
    shape `_citations()` returns, so downstream callers don't care which adapter
    produced it, and the cited_text-in-chunk invariant holds trivially. Out-of-range
    indices (the model naming a chunk that wasn't retrieved) are dropped."""
    text = " ".join(claim.statement for claim in grounded.claims)
    citations = [
        {
            "claim": claim.statement,
            "cited_text": hits[idx]["content"],
            "chunk_id": hits[idx]["id"],
            "title": hits[idx]["title"],
            "source": hits[idx]["source"],
        }
        for claim in grounded.claims
        for idx in claim.chunk_indices
        if 0 <= idx < len(hits)
    ]
    return text, citations


def answer(question: str, hits: list[dict],
           model: str = GEN_MODEL, system: str = SYSTEM_PROMPT,
           provider: str = GEN_PROVIDER) -> tuple[str, list[dict]]:
    """Answer over the retrieved chunks and return (answer_text, citations).

    Dispatches by `provider`: "anthropic" uses the native Citations API (the baseline
    path, unchanged); "openai-compat" reaches an OpenAI-compatible endpoint through
    instructor and rebuilds citations from structured output. `model`, `system`, and
    `provider` default to the production constants; the eval runner overrides them to
    test a config (evals/run.py)."""
    if provider == "anthropic":
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=_messages(question, hits),
        )
        text = "".join(block.text for block in response.content)
        return text, _citations(response.content, hits)

    # api_key is OPENROUTER_API_KEY passed as the OpenAI key against OpenRouter's base
    # URL — instructor's generic openai provider would otherwise read OPENAI_API_KEY,
    # which isn't the key this path uses. Mode.JSON, not the default TOOLS: DeepSeek on
    # OpenRouter doesn't reliably emit tool calls for the schema (it returns the fields as
    # prose), so we ask for JSON in the content instead — the mode instructor recommends
    # for OpenAI-compatible models with flaky function-calling.
    client = instructor.from_provider(
        f"openai/{model}",
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        mode=instructor.Mode.JSON,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _openai_user_content(question, hits)},
    ]
    # OpenRouter load-balances across backends; under batch load some occasionally
    # truncate the response (finish_reason=length) even though the same request succeeds
    # on a retry. instructor's max_retries only covers parse/validation errors, not a
    # length cutoff, so retry that case ourselves.
    for attempt in range(3):
        try:
            grounded = client.create(
                response_model=GroundedAnswer,
                max_tokens=STRUCTURED_MAX_TOKENS,
                messages=messages,
            )
            break
        except IncompleteOutputException:
            if attempt == 2:
                raise
    return _chunk_citations(grounded, hits)


def answer_stream(question: str, hits: list[dict],
                  model: str = GEN_MODEL, provider: str = GEN_PROVIDER):
    """Yield the answer incrementally, then one citation record per source.

    Both adapters keep the same text-first / citations-last contract. The Anthropic
    path streams token by token; the OpenAI-compatible path can't token-stream a
    structured response, so it returns the prose in one text event then the citations
    (citations-at-end — live streaming isn't required). Each item is yielded as
    {"type": "text"|"citation", ...}."""
    if provider == "anthropic":
        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=_messages(question, hits),
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "text", "text": text}
            final = stream.get_final_message()
        for cite in _citations(final.content, hits):
            yield {"type": "citation", **cite}
        return

    text, citations = answer(question, hits, model=model, provider=provider)
    yield {"type": "text", "text": text}
    for cite in citations:
        yield {"type": "citation", **cite}


def _overlap(prefix_lines: list[str], next_lines: list[str]) -> int:
    """How many trailing lines of prefix_lines equal the leading lines of next_lines.

    Adjacent chunks share a CHUNK_OVERLAP-sized run of whole lines (see chunk.py),
    so this is how many lines to drop from next_lines to avoid repeating them."""
    for k in range(min(len(prefix_lines), len(next_lines)), 0, -1):
        if prefix_lines[-k:] == next_lines[:k]:
            return k
    return 0


def _stitch(rows: list[dict], target_index: int) -> tuple[str, str, str]:
    """Join consecutive chunks into (before, chunk, after), dropping overlap.

    The target chunk is kept whole so a citation's quote is always findable inside
    `chunk`; the duplicated boundary lines are trimmed from the neighbours instead."""
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

    Returns the chunk stitched back together with `window` chunks of context on each
    side, plus the document title and the chunk's section (its nearest heading, or
    the document's section if the chunk has none) and position. The frontend shows
    `before`/`after` as muted context and highlights `chunk`."""
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
