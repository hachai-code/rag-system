"""The agents' Q&A long-term memory: a semantic cache over a plain pgvector table, shared
by the web agent (/agent/stream) and the deep agent (/ask/agent*).

Each completed run stores its question (Voyage-embedded), answer, and sources in
`qa_memory`; a later run whose question is close enough (cosine similarity) is shown the
past Q&As as leads. The same table backs the read-only /qa views.

The question is embedded with input_type="document" on both write and lookup so stored
questions and query questions share one space (Voyage's query/document asymmetry isn't
useful for question-to-question matching). Table lives in db/migrations/0004.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

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


@dataclass
class QAMatch:
    """The run-start cache lookup. `block` feeds the agent's prompt; `top_score` gates the
    end-of-run write (skip near-duplicates); `embedding` is the question's vector, reused by
    save_qa_record so a run pays for one Voyage call, not two."""

    block: str
    top_score: float
    embedding: list[float]


def lookup_similar_qa(
    conn: psycopg.Connection, question: str, user_id: UUID, limit: int = 3
) -> QAMatch:
    """The top-N similar past Q&As from the asking user's own cache, formatted for the
    prompt (block "" and top_score 0.0 when the cache has nothing)."""
    embedding = _embed(question)
    rows = conn.execute(
        """
        SELECT value, 1 - (embedding <=> %(emb)s::vector) AS score
        FROM qa_memory
        WHERE user_id = %(uid)s
        ORDER BY embedding <=> %(emb)s::vector
        LIMIT %(limit)s
        """,
        {"emb": embedding, "uid": user_id, "limit": limit},
    ).fetchall()
    if not rows:
        return QAMatch("", 0.0, embedding)
    block = "\n\n".join(
        f"### Past Q (similarity {r['score']:.2f}): {r['value']['question']}\n"
        f"{r['value']['answer'][:QA_ANSWER_CHARS]}\n"
        f"Web sources: {', '.join(r['value'].get('web_urls', [])) or 'none'}"
        for r in rows
    )
    return QAMatch(block, rows[0]["score"], embedding)


def save_qa_record(
    conn: psycopg.Connection,
    question: str,
    answer: str,
    corpus_sources: list[dict],
    web_urls: list[str],
    match: QAMatch,
    user_id: UUID,
    created_at: datetime | None = None,
) -> None:
    """One Q&A cache record per completed run, owned by `user_id` and semantically indexed
    on the question. Skipped when the run-start lookup found a near-duplicate cached
    question: the answer leaned on that record, so writing it again would only duplicate.
    `created_at` defaults to now; the Langfuse backfill passes the original trace time."""
    if match.top_score >= QA_DEDUP_SCORE:
        return
    value = {
        "question": question,
        "answer": answer,
        "corpus_sources": corpus_sources,
        "web_urls": web_urls,
        "research_files": {},  # the virtual FS is gone; findings return in-band now
    }
    conn.execute(
        "INSERT INTO qa_memory (key, question, value, embedding, user_id, created_at)"
        " VALUES (%s, %s, %s, %s::vector, %s, COALESCE(%s, now()))",
        (
            uuid4().hex,
            question,
            psycopg.types.json.Jsonb(value),
            match.embedding,
            user_id,
            created_at,
        ),
    )
    conn.commit()


def list_memories(conn: psycopg.Connection, user_id: UUID, limit: int = 200) -> list[dict]:
    """The user's own stored Q&A memories, newest first — the /qa list view."""
    return conn.execute(
        "SELECT key, question, created_at FROM qa_memory"
        " WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
        (user_id, limit),
        # ponytail: hard cap, paginate if the cache outgrows it
    ).fetchall()


def delete_memory(conn: psycopg.Connection, key: str, user_id: UUID) -> bool:
    """Delete one of the user's stored Q&A memories. True if a row was removed, False when
    the key is unknown or owned by someone else."""
    deleted = conn.execute(
        "DELETE FROM qa_memory WHERE key = %s AND user_id = %s", (key, user_id)
    ).rowcount
    conn.commit()
    return deleted > 0


def get_memory(conn: psycopg.Connection, key: str, user_id: UUID) -> dict | None:
    """One of the user's stored Q&A memories in full — the /qa/{key} detail view. None when
    unknown or owned by someone else."""
    row = conn.execute(
        "SELECT question, value, created_at FROM qa_memory WHERE key = %s AND user_id = %s",
        (key, user_id),
    ).fetchone()
    if row is None:
        return None
    return {"key": key, "created_at": row["created_at"], **row["value"]}
