"""Minimal RAG over the innerdance corpus: retrieve, then (later) ask Claude.

This is the query side of the pipeline that ingest -> chunk -> embed built.
Run `uv run rag.py` to see the top-k chunks for a sample question.
"""

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
        SELECT d.title, d.source, c.content,
               c.embedding <=> %(emb)s::vector AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %(emb)s::vector
        LIMIT %(k)s
        """,
        {"emb": embedding, "k": k},
    ).fetchall()


if __name__ == "__main__":
    question = "What is the relationship between epilepsy and spiritual experience?"
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = search(conn, question)

    print(f"Q: {question}\n")
    for i, hit in enumerate(hits, 1):
        preview = " ".join(hit["content"].split())[:90]
        print(f"{i}. [{hit['distance']:.4f}] {hit['title']}")
        print(f"   {preview}...\n")
