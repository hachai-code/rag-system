"""Build the index end to end: ingest -> chunk -> embed -> store, in that order.

Run: uv run python -m rag.pipeline  (needs the Postgres container up and VOYAGE_API_KEY)
"""

import logging
import os

import psycopg
import voyageai
from ai_utils import UsageTracker
from pgvector.psycopg import register_vector

from .db import DB_URL
from .indexing.chunk import chunk_corpus
from .indexing.embed import VOYAGE_MODEL, embed_batches, store_chunks, upsert_documents
from .indexing.ingest import CORPUS_ROOT, load_corpus


def build_index() -> None:
    documents = load_corpus(CORPUS_ROOT)
    chunks = chunk_corpus(documents)
    print(f"Embedding {len(chunks)} chunks with {VOYAGE_MODEL} ...")

    client = voyageai.Client()
    tracker = UsageTracker()
    # Contextual prefix (section + title) on each chunk's embedded text — improves
    # retrieval (README "Pipeline order"). The stored content stays raw.
    texts = [f"{c.section} — {c.title[:60]}\n\n{c.content}" for c in chunks]
    embeddings = embed_batches(client, texts, tracker)

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        doc_ids = upsert_documents(conn, documents)
        store_chunks(conn, chunks, embeddings, doc_ids)

    print(f"Stored {len(chunks)} chunks across {len(documents)} documents.")
    print(f"Voyage usage: {tracker.summary()}")


def main() -> None:
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY is not set. Add it to .env or export it.")

    # Surface ai_utils' per-call cost logs while keeping third-party libs quiet.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("ai_utils").setLevel(logging.INFO)

    build_index()


if __name__ == "__main__":
    main()
