"""Embed every chunk with Voyage and store the vectors in pgvector.

Run: uv run embed.py
Requires: the Postgres container up (docker compose up -d) and VOYAGE_API_KEY
set in the environment or a .env file.

This is idempotent — re-running re-embeds and replaces the existing rows rather
than duplicating them.
"""

import datetime
import logging
import os
import time

import psycopg
import voyageai
from ai_utils import UsageTracker
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from chunk import Chunk, chunk_corpus
from ingest import CORPUS_ROOT, Document, load_corpus
from rag import DB_URL  # local default, or DATABASE_URL if set

load_dotenv()

VOYAGE_MODEL = "voyage-4"  # same price as 3.5, newer, includes 200M free tokens
EMBED_DIM = 1024   # must match the VECTOR(1024) column in db/init.sql
BATCH_SIZE = 128   # texts per request; limits are 1000 texts / 320K tokens

# Transient failures worth retrying. Auth/bad-request errors are excluded —
# retrying those just wastes time.
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
    """Embed texts in batches. input_type='document' tunes the vectors for being
    the searched corpus (queries are embedded with input_type='query')."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        result = _embed_with_retry(client, batch)
        embeddings.extend(result.embeddings)
        tracker.record(VOYAGE_MODEL, result.total_tokens)
        print(f"  embedded {start + len(batch)}/{len(texts)}", flush=True)
    return embeddings


def _embed_with_retry(client: voyageai.Client, batch: list[str], attempts: int = 5):
    """Voyage occasionally returns 503 (overloaded). Back off and retry rather
    than losing the whole run to a transient blip."""
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
        (doc_ids[c.source], c.chunk_index, c.content, embedding, Jsonb({"heading": c.heading}))
        for c, embedding in zip(chunks, embeddings)
    ]
    conn.cursor().executemany(
        """
        INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """,
        rows,
    )


def main() -> None:
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY is not set. Add it to .env or export it.")

    # Surface ai_utils' per-call cost logs while keeping third-party libs quiet.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("ai_utils").setLevel(logging.INFO)

    documents = load_corpus(CORPUS_ROOT)
    chunks = chunk_corpus(documents)
    print(f"Embedding {len(chunks)} chunks with {VOYAGE_MODEL} ...")

    client = voyageai.Client()
    tracker = UsageTracker()
    # Contextual prefix (section + title) prepended to each chunk's *embedded*
    # text — measurably improves retrieval (see chunking-experiments.md). The
    # stored content stays raw; only the vector sees the prefix.
    texts = [f"{c.section} — {c.title[:60]}\n\n{c.content}" for c in chunks]
    embeddings = embed_batches(client, texts, tracker)

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        doc_ids = upsert_documents(conn, documents)
        store_chunks(conn, chunks, embeddings, doc_ids)

    print(f"Stored {len(chunks)} chunks across {len(documents)} documents.")
    print(f"Voyage usage: {tracker.summary()}")


if __name__ == "__main__":
    main()
