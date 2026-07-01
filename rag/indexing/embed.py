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

from .chunk import Chunk
from .ingest import Document

VOYAGE_MODEL = "voyage-4"  # same price as 3.5, newer, includes 200M free tokens
EMBED_DIM = 1024   # must match the VECTOR(1024) column in db/init.sql
BATCH_SIZE = 128   # texts per request; limits are 1000 texts / 320K tokens

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
