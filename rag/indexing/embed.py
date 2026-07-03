"""Stage 3: embed chunks with Voyage and store them in pgvector.

Functions only — `rag.pipeline.build_index` runs the full ingest -> chunk -> embed
sequence. Storing is idempotent: it replaces a document's chunks rather than
duplicating them.
"""

import datetime
import time

import psycopg
import voyageai
from ai_utils import UsageTracker
from psycopg.types.json import Jsonb

from ..config import CONFIG
from ..query.answer import complete
from .chunk import Chunk
from .ingest import Document

VOYAGE_MODEL = CONFIG.voyage_model
EMBED_DIM = 1024   # must match the VECTOR(1024) column in db/migrations/
BATCH_SIZE = 128   # texts per request; limits are 1000 texts / 320K tokens

# HyPE: cheap model that writes the hypothetical questions, and how many per chunk.
HYPE_GEN_MODEL = CONFIG.gen_models["flash"]
N_HYPE_QUESTIONS = 4

# Transient failures worth retrying; auth/bad-request errors are excluded.
_TRANSIENT_ERRORS = (
    voyageai.error.ServiceUnavailableError,
    voyageai.error.ServerError,
    voyageai.error.APIConnectionError,
    voyageai.error.Timeout,
    voyageai.error.RateLimitError,
)


def embed_batches(
    client: voyageai.Client, texts: list[str], tracker: UsageTracker
) -> list[list[float]]:
    """Embed texts in batches. input_type='document' tunes the vectors for being the
    searched corpus (queries use input_type='query')."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        result = _embed_with_retry(client, batch)
        embeddings.extend(result.embeddings)
        tracker.record(VOYAGE_MODEL, result.total_tokens)
        print(f"  embedded {start + len(batch)}/{len(texts)}", flush=True)
    return embeddings


def _embed_with_retry(client: voyageai.Client, batch: list[str], attempts: int = 5):
    """Back off and retry on transient Voyage errors (e.g. 503 overloaded)."""
    for attempt in range(attempts):
        try:
            return client.embed(
                batch,
                model=VOYAGE_MODEL,
                input_type="document",
                output_dimension=EMBED_DIM,
            )
        except _TRANSIENT_ERRORS as exc:
            if attempt == attempts - 1:
                raise
            wait = 2**attempt
            print(f"  Voyage error ({type(exc).__name__}); retrying in {wait}s ...", flush=True)
            time.sleep(wait)


def upsert_documents(conn: psycopg.Connection, documents: list[Document]) -> dict[str, int]:
    """Insert (or update) each document and return a source -> id map."""
    ids = {}
    for doc in documents:
        row = conn.execute(
            """
            INSERT INTO documents (source, title, section, doc_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title
            RETURNING id
            """,
            (doc.source, doc.title, doc.section, datetime.date.fromisoformat(doc.date)),
        ).fetchone()
        ids[doc.source] = row[0]
    return ids


def store_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    doc_ids: dict[str, int],
) -> None:
    """Replace all chunks for the embedded documents with the fresh ones."""
    conn.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (list(doc_ids.values()),))
    rows = [
        (
            doc_ids[c.source], c.chunk_index, c.content, embedding,
            Jsonb({
                "heading": c.heading,
                "start": c.start,
                "end": c.end,
                "speakers": list(c.speakers) if c.speakers else None,
                "primary_speaker": c.primary_speaker,
            }),
        )
        for c, embedding in zip(chunks, embeddings)
    ]
    conn.cursor().executemany(
        """
        INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """,
        rows,
    )


HYPE_PROMPT = (
    "Write {n} distinct questions that this passage from the innerdance corpus directly "
    "answers — the kind of questions a reader would ask. One per line, no numbering or "
    "commentary.\n\nPassage:\n{content}"
)


def hypothetical_questions(content: str, n: int = N_HYPE_QUESTIONS, model: str = HYPE_GEN_MODEL) -> list[str]:
    """N hypothetical questions the chunk answers (HyPE), via the cheap flash model."""
    text = complete(HYPE_PROMPT.format(n=n, content=content), model)
    return [line.strip() for line in text.splitlines() if line.strip()][:n]


def populate_hype(
    conn: psycopg.Connection,
    client: voyageai.Client,
    tracker: UsageTracker,
    chunk_ids: list[int] | None = None,
    n: int = N_HYPE_QUESTIONS,
) -> int:
    """Generate hypothetical questions per chunk, embed them, and (re)store them in
    chunk_questions. A separate, opt-in step: the chunks table is never touched, so the
    served `content`/`embedding` stay the raw chunk. Idempotent — a chunk's old questions
    are cleared before reinsert. Pass `chunk_ids` to run over a subset for testing."""
    if chunk_ids is None:
        rows = conn.execute("SELECT id, content FROM chunks ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content FROM chunks WHERE id = ANY(%s) ORDER BY id", (list(chunk_ids),)
        ).fetchall()

    ids: list[int] = []
    questions: list[str] = []
    for chunk_id, content in rows:
        for question in hypothetical_questions(content, n):
            ids.append(chunk_id)
            questions.append(question)
        print(f"  questions {len(ids)} (chunk {chunk_id})", flush=True)

    embeddings = embed_batches(client, questions, tracker)

    conn.execute("DELETE FROM chunk_questions WHERE chunk_id = ANY(%s)", ([r[0] for r in rows],))
    conn.cursor().executemany(
        "INSERT INTO chunk_questions (chunk_id, question, embedding) VALUES (%s, %s, %s)",
        list(zip(ids, questions, embeddings)),
    )
    return len(questions)


def main() -> None:
    """Populate chunk_questions for HyPE retrieval. Run after the main index is built:

        uv run python -m rag.indexing.embed            # whole corpus
        uv run python -m rag.indexing.embed --limit 20 # first 20 chunks (testing)
    """
    import argparse

    from pgvector.psycopg import register_vector

    from ..db import DB_URL

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N chunks (quick test)")
    args = ap.parse_args()

    client = voyageai.Client()
    tracker = UsageTracker()
    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        if args.limit:
            chunk_ids = [r[0] for r in conn.execute(
                "SELECT id FROM chunks ORDER BY id LIMIT %s", (args.limit,)
            ).fetchall()]
        else:
            chunk_ids = None
        n = populate_hype(conn, client, tracker, chunk_ids)
    print(f"Stored {n} hypothetical questions. Voyage usage: {tracker.summary()}")


if __name__ == "__main__":
    main()
