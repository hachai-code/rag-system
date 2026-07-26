"""Click-through: reconstruct where a cited chunk sits in its document.

Read-side custom SQL behind `/source/{chunk_id}`, kept apart from retrieval ranking —
this runs after an answer is cited, not on the retrieve -> answer path.
"""

import psycopg

from ..db import Hit
from .retrieve import SOURCE_WINDOW


def _overlap(prefix_lines: list[str], next_lines: list[str]) -> int:
    """How many trailing lines of prefix_lines equal the leading lines of next_lines.

    Adjacent chunks share a CHUNK_OVERLAP-sized run of whole lines (see chunk.py)."""
    for k in range(min(len(prefix_lines), len(next_lines)), 0, -1):
        if prefix_lines[-k:] == next_lines[:k]:
            return k
    return 0


def _stitch(rows: list[Hit], target_index: int) -> tuple[str, str, str]:
    """Join consecutive chunks into (before, chunk, after), dropping overlap.

    The target chunk is kept whole so a citation's quote stays findable inside it."""
    before: list[str] = []
    chunk: list[str] = []
    after: list[str] = []
    for row in rows:
        lines = row["content"].split("\n")
        if row["chunk_index"] < target_index:
            before.extend(lines[_overlap(before, lines) :])
        elif row["chunk_index"] == target_index:
            chunk = lines
            del before[len(before) - _overlap(before, lines) :]  # trim before's tail
        else:
            after.extend(lines[_overlap(chunk + after, lines) :])
    return "\n".join(before), "\n".join(chunk), "\n".join(after)


def source_passage(conn: psycopg.Connection, chunk_id: int, window: int = SOURCE_WINDOW) -> dict:
    """Reconstruct where a cited chunk sits in its document, for click-through.

    Returns the chunk stitched back with `window` chunks of context on each side, plus
    the document title, the chunk's section (nearest heading, or the document's), and
    its position."""
    target = conn.execute(
        "SELECT document_id, chunk_index FROM chunks WHERE id = %s", (chunk_id,)
    ).fetchone()
    doc = conn.execute(
        "SELECT title, section FROM documents WHERE id = %s", (target["document_id"],)
    ).fetchone()
    n_chunks = conn.execute(
        "SELECT count(*) AS n FROM chunks WHERE document_id = %s", (target["document_id"],)
    ).fetchone()["n"]
    rows = conn.execute(
        """SELECT chunk_index, content, metadata->>'heading' AS heading
           FROM chunks
           WHERE document_id = %s AND chunk_index BETWEEN %s AND %s
           ORDER BY chunk_index""",
        (target["document_id"], target["chunk_index"] - window, target["chunk_index"] + window),
    ).fetchall()

    before, chunk, after = _stitch(rows, target["chunk_index"])
    heading = next((r["heading"] for r in rows if r["chunk_index"] == target["chunk_index"]), None)
    return {
        "title": doc["title"],
        "section": heading or doc["section"],
        "chunk_index": target["chunk_index"],
        "n_chunks": n_chunks,
        "before": before,
        "chunk": chunk,
        "after": after,
    }
