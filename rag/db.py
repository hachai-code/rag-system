"""Shared Postgres access: the DB_URL and a configured-connection helper."""

import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")


def connect() -> psycopg.Connection:
    """A dict-row connection with pgvector registered — the shape query code expects."""
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    register_vector(conn)
    return conn
