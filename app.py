"""FastAPI wrapper around the RAG pipeline: POST a question, get a grounded answer.

Run: uv run fastapi dev app.py
Then: curl -X POST localhost:8000/ask -H 'content-type: application/json' \
        -d '{"question": "..."}'
"""

import psycopg
from fastapi import FastAPI
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from pydantic import BaseModel

from rag import DB_URL, answer, search

app = FastAPI(title="innerdance RAG")


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# Sync `def` (not async): the Voyage/Claude/psycopg calls block, so FastAPI runs
# this in a threadpool instead of stalling the event loop.
@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = search(conn, request.question)
    return AskResponse(
        answer=answer(request.question, hits),
        sources=[
            Source(title=h["title"], source=h["source"], distance=h["distance"])
            for h in hits
        ],
    )
