"""Shared Postgres access: the DB_URL, a configured-connection helper, and the row shape."""

import os
from typing import NotRequired, TypedDict

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")

EMBED_DIM = 1024  # must match the VECTOR(1024) columns in db/migrations/


class Hit(TypedDict):
    """One retrieved chunk row — the currency between retrieval and generation.

    `distance` (cosine) is set by the vector stages and None on parent-document
    neighbours; keyword hits carry `rank` (ts_rank) instead."""

    id: int
    title: str
    source: str
    content: str
    distance: NotRequired[float | None]
    rank: NotRequired[float]


def connect() -> psycopg.Connection:
    """A dict-row connection with pgvector registered — the shape query code expects."""
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    register_vector(conn)
    return conn
