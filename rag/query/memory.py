"""The deep agent's Q&A long-term memory: a semantic cache over a plain pgvector table.

Each completed run stores its question (Voyage-embedded), answer, and sources in
`qa_memory`; a later run whose question is close enough (cosine similarity) is shown the
past Q&As as leads. The same table backs the read-only /qa views.

The question is embedded with input_type="document" on both write and lookup so stored
questions and query questions share one space (Voyage's query/document asymmetry isn't
useful for question-to-question matching). Table lives in db/migrations/0004.
"""

from uuid import uuid4

import psycopg

from ..clients import voyage_client
from ..db import EMBED_DIM
from .retrieve import VOYAGE_MODEL

QA_ANSWER_CHARS = 2000  # cap per past answer so 3 hits don't dominate the prompt
QA_DEDUP_SCORE = 0.9  # ponytail: fixed threshold, tune if dupes or collisions appear


def _embed(text: str) -> list[float]:
    """Embed a question for the cache. input_type="document" on both sides keeps stored
    questions and lookup queries in one consistent space."""
    return (
        voyage_client()
        .embed([text], model=VOYAGE_MODEL, input_type="document", output_dimension=EMBED_DIM)
        .embeddings[0]
    )


def lookup_similar_qa(conn: psycopg.Connection, question: str, limit: int = 3) -> tuple[str, float]:
    """(formatted top-N similar past Q&As, top cosine similarity) — ("", 0.0) when the
    cache has nothing. The block feeds the corpus agent's prompt; the top score gates the
    end-of-run cache write in save_qa_record (skip near-duplicates)."""
    rows = conn.execute(
        """
        SELECT value, 1 - (embedding <=> %(emb)s::vector) AS score
        FROM qa_memory
        ORDER BY embedding <=> %(emb)s::vector
        LIMIT %(limit)s
        """,
        {"emb": _embed(question), "limit": limit},
    ).fetchall()
    if not rows:
        return "", 0.0
    block = "\n\n".join(
        f"### Past Q (similarity {r['score']:.2f}): {r['value']['question']}\n"
        f"{r['value']['answer'][:QA_ANSWER_CHARS]}\n"
        f"Web sources: {', '.join(r['value'].get('web_urls', [])) or 'none'}"
        for r in rows
    )
    return block, rows[0]["score"]


def save_qa_record(
    conn: psycopg.Connection,
    question: str,
    answer: str,
    corpus_sources: list[dict],
    web_urls: list[str],
    top_score: float,
) -> None:
    """One Q&A cache record per completed run, semantically indexed on the question.
    Skipped when the run-start lookup found a near-duplicate cached question: the answer
    leaned on that record, so writing it again would only duplicate."""
    if top_score >= QA_DEDUP_SCORE:
        return
    value = {
        "question": question,
        "answer": answer,
        "corpus_sources": corpus_sources,
        "web_urls": web_urls,
        "research_files": {},  # the virtual FS is gone; findings return in-band now
    }
    conn.execute(
        "INSERT INTO qa_memory (key, question, value, embedding) VALUES (%s, %s, %s, %s::vector)",
        (uuid4().hex, question, psycopg.types.json.Jsonb(value), _embed(question)),
    )
    conn.commit()


def list_memories(conn: psycopg.Connection, limit: int = 200) -> list[dict]:
    """The stored Q&A memories, newest first — the /qa list view."""
    return conn.execute(
        "SELECT key, question, created_at FROM qa_memory ORDER BY created_at DESC LIMIT %s",
        (limit,),
        # ponytail: hard cap, paginate if the cache outgrows it
    ).fetchall()


def get_memory(conn: psycopg.Connection, key: str) -> dict | None:
    """One stored Q&A memory in full — the /qa/{key} detail view. None when unknown."""
    row = conn.execute(
        "SELECT question, value, created_at FROM qa_memory WHERE key = %s", (key,)
    ).fetchone()
    if row is None:
        return None
    return {"key": key, "created_at": row["created_at"], **row["value"]}
