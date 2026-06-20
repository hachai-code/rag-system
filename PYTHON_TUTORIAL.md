# Learn Python by Reading a Real RAG System

This tutorial teaches Python using the actual code in this repository — a
working Retrieval-Augmented Generation (RAG) system that loads a corpus,
chunks it, embeds it into a vector database, and answers questions with Claude.

Every example below is real code from this project (lightly trimmed). The idea
is to learn Python the way you'll actually use it: not toy snippets, but the
patterns that show up in production code. Each section names the file it comes
from so you can open it and read the surrounding context.

**How to use this:** read top to bottom if you're newer to Python, or jump to a
topic. Each section has a *Concept*, the *Code from the repo*, and a *Why it
works this way* explanation.

---

## Table of contents

1. [The shape of a Python file](#1-the-shape-of-a-python-file)
2. [Functions and type hints](#2-functions-and-type-hints)
3. [Dataclasses: structured data](#3-dataclasses-structured-data)
4. [Strings, f-strings, and formatting](#4-strings-f-strings-and-formatting)
5. [Lists, dicts, and comprehensions](#5-lists-dicts-and-comprehensions)
6. [Control flow and iteration tricks](#6-control-flow-and-iteration-tricks)
7. [Working with files and paths](#7-working-with-files-and-paths)
8. [Dispatch tables: dicts instead of if-chains](#8-dispatch-tables-dicts-instead-of-if-chains)
9. [Closures and `nonlocal`](#9-closures-and-nonlocal)
10. [Generators and `yield`](#10-generators-and-yield)
11. [Decorators and caching](#11-decorators-and-caching)
12. [Context managers (`with`)](#12-context-managers-with)
13. [Exceptions and retries](#13-exceptions-and-retries)
14. [Modules, imports, and environment config](#14-modules-imports-and-environment-config)
15. [Pydantic and FastAPI: validation at the edge](#15-pydantic-and-fastapi-validation-at-the-edge)
16. [Putting it together: the full pipeline](#16-putting-it-together-the-full-pipeline)

---

## 1. The shape of a Python file

**Concept.** Every Python script has a recognizable skeleton: a module
docstring at the top, imports, then definitions, and often a `main()` guarded by
`if __name__ == "__main__":`.

**Code** (`main.py`, the whole file):

```python
def main():
    print("Hello from rag-system!")


if __name__ == "__main__":
    main()
```

**Why it works this way.** When you run `python main.py`, Python sets the
special variable `__name__` to `"__main__"`, so the `if` block fires and calls
`main()`. But when another file does `import main`, `__name__` is `"main"`
instead, so the block is skipped — importing the module doesn't accidentally run
it. This is why almost every file in this repo ends with that exact guard
(`ingest.py`, `chunk.py`, `embed.py`, `rag.py`).

The triple-quoted string at the very top of a file is a **module docstring** —
documentation that tools and `help()` can read. From `ingest.py`:

```python
"""Load the innerdance corpus into clean text + metadata, then print its shape.

Run: uv run ingest.py

This is the first stage of the RAG pipeline. It does NOT chunk or embed yet ...
"""
```

---

## 2. Functions and type hints

**Concept.** Functions are defined with `def`. Modern Python annotates
parameters and return values with **type hints**. They don't change how the code
runs — Python doesn't enforce them at runtime — but they document intent and let
editors and type checkers catch mistakes.

**Code** (`ingest.py`):

```python
def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))
```

Read this as: "`count_tokens` takes a `text` that is a `str`, and returns an
`int`." Here's one with several typed parameters and defaults (`chunk.py`):

```python
def chunk_document(
    doc: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Greedily pack lines into ~chunk_size-token chunks with ~overlap carryover."""
    ...
```

**Why it works this way.**

- `chunk_size: int = CHUNK_SIZE` gives the parameter a **default value**. Callers
  can write `chunk_document(doc)` and get the default, or override it with
  `chunk_document(doc, chunk_size=128)`.
- `-> list[Chunk]` says the function returns a list of `Chunk` objects. Modern
  Python (3.9+) lets you write `list[Chunk]` directly instead of importing
  `List` from `typing`.
- `list[float]` (used all over `embed.py`) means "a list of floats" — exactly
  what an embedding vector is.

A type hint can also say "this or `None`" with `|` (`chunk.py`):

```python
heading: str | None  # nearest preceding heading line, if any
```

`str | None` means "a string, or nothing." This is the modern replacement for
`Optional[str]`.

---

## 3. Dataclasses: structured data

**Concept.** When you need a record with named fields, a `@dataclass` gives you
one with almost no boilerplate. You declare the fields and their types; Python
generates the `__init__`, a readable `__repr__`, equality, and more.

**Code** (`ingest.py`):

```python
from dataclasses import dataclass


@dataclass
class Document:
    source: str    # path relative to the corpus root
    title: str     # human-readable name (from the filename)
    date: str      # ISO date the file was last modified
    section: str   # top-level corpus folder, e.g. "Maia" or "Documents"
    text: str      # cleaned plain text
    n_tokens: int  # estimated token count of `text`
```

Now you can create and use one naturally:

```python
doc = Document(
    source="Maia/episode1.rtf",
    title="episode1",
    date="2026-01-15",
    section="Maia",
    text="...",
    n_tokens=4200,
)
print(doc.title)      # "episode1"
print(doc.n_tokens)   # 4200
```

**Why it works this way.** The `@dataclass` line is a **decorator** (more on
those in §11) that rewrites the class for you. Without it you'd hand-write an
`__init__` that assigns all six fields. The `Chunk` class in `chunk.py` works
the same way. Dataclasses are the right tool when the data has a fixed, known
shape and you want attribute access (`doc.title`) rather than dictionary keys
(`doc["title"]`).

---

## 4. Strings, f-strings, and formatting

**Concept.** An **f-string** (prefix `f"..."`) lets you embed expressions
directly inside a string with `{...}`. It's the standard way to build strings in
modern Python.

**Code** (`embed.py`):

```python
texts = [f"{c.section} — {c.title[:60]}\n\n{c.content}" for c in chunks]
```

This builds, for each chunk, a string like:

```
Maia — Episode 1: The Architecture

<the chunk's content>
```

Two things are happening inside the braces:

- `c.title[:60]` is **slicing** — take the first 60 characters of the title.
- `\n\n` is an escape sequence — two newlines, producing the blank line.

**Format specifiers** go after a colon and control how a value is rendered
(`ingest.py`):

```python
print(f"Total tokens:  {total_tokens:,} (~{total_tokens / 1000:.0f}k)")
```

- `{total_tokens:,}` inserts thousands separators → `1,234,567`.
- `{... :.0f}` formats a float with zero decimal places → `1235k`.

And alignment, used to print aligned tables (`chunk.py`):

```python
print(f"  {len(doc_chunks):>4}  {source}")
```

`:>4` right-aligns the number in a field 4 characters wide. There's also `:<40`
(left-align in 40 chars) used elsewhere to lay out bar charts.

You can format with `!r` to get the `repr()` (quoted, debug form) of a value
(`chunk.py`):

```python
print(f"... heading={c.heading!r}")   # heading='PART ONE' rather than heading=PART ONE
```

---

## 5. Lists, dicts, and comprehensions

**Concept.** A **comprehension** builds a list, dict, or set from an iterable in
one expression. It replaces the common "make an empty list, loop, append"
pattern.

**Filtering** with a comprehension (`chunk.py`):

```python
lines = [line for line in doc.text.split("\n") if line.strip()]
```

Read it right-to-left of the `for`: "for each `line` in the split text, keep it
**if** `line.strip()` is truthy (i.e. the line isn't blank)." The equivalent
long form would be:

```python
lines = []
for line in doc.text.split("\n"):
    if line.strip():
        lines.append(line)
```

**Transforming** each item (`rag.py`):

```python
text = "".join(block.text for block in response.content)
```

This is a **generator expression** (no brackets) feeding `join` — it pulls the
`.text` out of each block and concatenates them.

**Dict comprehension** (`rag.py`):

```python
by_id = {hit["id"]: hit["content"] for hit in hits}
```

This builds a lookup table mapping each chunk's id to its content, so later code
can do `by_id[some_id]` in O(1).

**A list of dicts** built from objects (`rag.py`):

```python
return [
    {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": hit["content"]},
        "title": hit["title"],
        "citations": {"enabled": True},
    }
    for hit in hits
]
```

**Why it works this way.** Comprehensions are both shorter and usually faster
than the manual loop. The rule of thumb: use a comprehension when you're
*building a collection*; use a regular `for` loop when you're doing side effects
(like printing) or the logic is too complex to read on one line.

---

## 6. Control flow and iteration tricks

**`enumerate`** gives you the index alongside each item. Note the `start=1` to
count from 1 instead of 0 (`rag.py`):

```python
for rank, hit in enumerate(hits, 1):
    scores[hit["id"]] = scores.get(hit["id"], 0.0) + weight / (RRF_K + rank)
```

**`dict.get` with a default** — `scores.get(hit["id"], 0.0)` returns the current
score if the key exists, or `0.0` if it's the first time we've seen that id. This
avoids a `KeyError` and the need to check membership first.

**`zip`** walks two lists in lockstep (`embed.py`):

```python
rows = [
    (doc_ids[c.source], c.chunk_index, c.content, embedding, Jsonb({"heading": c.heading}))
    for c, embedding in zip(chunks, embeddings)
]
```

Each iteration pairs the *nth* chunk with the *nth* embedding.

**`reversed`** iterates back-to-front (`chunk.py`):

```python
for line, n in reversed(lines):
    if kept and total + n > overlap:
        break
    kept.append((line, n))
    total += n
```

**`next(...)` with a generator and a default** finds the first match, or returns
a fallback if there is none (`rag.py`):

```python
heading = next(
    (r["heading"] for r in rows if r["chunk_index"] == target["chunk_index"]), None
)
```

This is the idiomatic "find first item matching a condition" — no loop with a
flag variable needed. The second argument (`None`) is what you get if nothing
matches.

**`sorted` with a `key`** sorts by a computed value. Here, by a dict's own
values, descending (`rag.py`):

```python
top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
```

`scores.get` is passed as the key function, so each id sorts by its score;
`reverse=True` puts the highest first; `[:k]` keeps the top k.

**Range with a step** to sample evenly (`chunk.py`):

```python
for c in chunks[::step][:count]:
    ...
```

`chunks[::step]` takes every `step`-th chunk (slice with a step), then `[:count]`
caps how many.

---

## 7. Working with files and paths

**Concept.** The `pathlib` module models filesystem paths as objects. You build
paths with the `/` operator and call methods on them instead of stitching
strings.

**Code** (`ingest.py`):

```python
from pathlib import Path

CORPUS_ROOT = Path.home() / "Documents" / "innerdance corpus"
```

`Path.home()` is your home directory; `/` joins path segments in a
cross-platform way (no manual `os.path.join` or worrying about slashes).

**Walking a directory tree and filtering** (`ingest.py`):

```python
def load_corpus(root: Path) -> list[Document]:
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTRACTORS:
            continue
        print(f"  loading {path.relative_to(root)} ...", flush=True)
        documents.append(load_document(path, root))
    return documents
```

Path methods used here:

- `root.rglob("*")` — recursively yield every path under `root`.
- `path.is_file()` — is this a file (not a directory)?
- `path.suffix` — the extension, like `".pdf"`; `.lower()` normalizes case.
- `path.relative_to(root)` — strip the root prefix for clean display.
- `path.stem` (in `load_document`) — filename without extension.

**`continue`** skips to the next loop iteration — here, anything that isn't a
file with a known extension is skipped early ("guard clause" style), which keeps
the rest of the loop body un-indented.

**Reading file contents** (`ingest.py`):

```python
def extract_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")
```

`read_text` reads the whole file as a string. `errors="ignore"` drops bytes that
aren't valid UTF-8 rather than crashing — pragmatic when ingesting messy
real-world files. There's also `path.read_bytes()` (used for HTML) when you want
raw bytes.

**Reading a file line by line into JSON records** (`evals/metrics.py`):

```python
rows = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
```

This is the JSONL pattern (one JSON object per line): read the file, split into
lines, skip blanks, parse each line.

---

## 8. Dispatch tables: dicts instead of if-chains

**Concept.** When behavior depends on a value, a dict mapping values to
functions is often cleaner than a long `if/elif` chain. Functions are
first-class objects in Python — you can store them in data structures and call
them later.

**Code** (`ingest.py`):

```python
EXTRACTORS = {
    ".rtf": extract_rtf,
    ".pdf": extract_pdf,
    ".epub": extract_epub,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_plain,
    ".txt": extract_plain,
}
```

Notice the values are **the functions themselves** — `extract_rtf`, not
`extract_rtf()`. There are no parentheses, so we're storing the function, not
calling it. Then to dispatch (`ingest.py`):

```python
def load_document(path: Path, root: Path) -> Document:
    raw = EXTRACTORS[path.suffix.lower()](path)
    ...
```

Read `EXTRACTORS[path.suffix.lower()](path)` in two steps:

1. `EXTRACTORS[path.suffix.lower()]` looks up the right function for this file's
   extension (e.g. for `.pdf` it returns `extract_pdf`).
2. `(path)` then **calls** that function with `path`.

**Why it works this way.** Adding a new file format is one line in the dict, no
edits to `load_document`. The same `.htm`/`.html` both pointing at
`extract_html` shows how two keys can share one handler. This pattern — a
"dispatch table" or "registry" — scales far better than:

```python
# the brittle alternative this avoids:
if suffix == ".rtf":
    raw = extract_rtf(path)
elif suffix == ".pdf":
    raw = extract_pdf(path)
elif ...
```

---

## 9. Closures and `nonlocal`

**Concept.** A function defined inside another function can read the outer
function's variables. To *reassign* them, it needs the `nonlocal` keyword. A
nested function that captures surrounding variables is called a **closure**.

**Code** (`chunk.py`, inside `chunk_document`):

```python
def chunk_document(doc, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    chunk_heading: str | None = None

    def flush() -> None:
        nonlocal current, current_tokens, chunk_heading
        content = "\n".join(line for line, _ in current)
        chunks.append(Chunk(..., content=content, n_tokens=current_tokens, ...))
        current, current_tokens = _tail_overlap(current, overlap)
        chunk_heading = heading

    for line in lines:
        ...
        if current and current_tokens + n > chunk_size:
            flush()
        ...
```

**Why it works this way.** `flush()` needs to *modify* `current`,
`current_tokens`, and `chunk_heading` — variables that belong to the enclosing
`chunk_document`. Without `nonlocal`, an assignment like `current_tokens = ...`
inside `flush` would create a brand-new local variable that vanishes when
`flush` returns, leaving the outer one untouched. `nonlocal` says "these names
refer to the enclosing function's variables — assign to *those*."

Note the asymmetry: `chunks.append(...)` does **not** need `nonlocal`, because
appending *mutates* the existing list rather than rebinding the name `chunks`.
You only need `nonlocal` (or `global`) when you **reassign** a name with `=`.

Also note the tuple unpacking in two places:

- `for line, _ in current` — each item is a `(line, n_tokens)` tuple; `_` is the
  conventional name for "a value I'm deliberately ignoring."
- `current, current_tokens = _tail_overlap(...)` — the function returns a tuple,
  and this unpacks it into two variables in one line.

---

## 10. Generators and `yield`

**Concept.** A function that uses `yield` instead of `return` is a
**generator**: calling it doesn't run the body, it returns a lazy iterator.
Each `yield` produces one value and pauses; execution resumes on the next
request. This is how you stream results without building a giant list in memory.

**Code** (`rag.py`):

```python
def answer_stream(question: str, hits: list[dict]):
    """Yield the answer incrementally, then one citation record per source."""
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=_messages(question, hits),
    ) as stream:
        for text in stream.text_stream:
            yield {"type": "text", "text": text}
        final = stream.get_final_message()
    for cite in _citations(final.content, hits):
        yield {"type": "citation", **cite}
```

The consumer drives it with a `for` loop (`app.py`):

```python
for event in answer_stream(body.question, hits):
    if event["type"] == "text":
        answer_text.append(event["text"])
    yield f"data: {json.dumps(event)}\n\n"
```

**Why it works this way.** Claude's answer arrives token by token. Instead of
waiting for the whole answer, `answer_stream` `yield`s each piece the moment it
arrives, so the UI can display text live. The function "remembers" where it
paused — including being inside the `with` block — and continues from there on
each iteration. After the text is exhausted it yields the citations.

The `**cite` in `{"type": "citation", **cite}` is **dict unpacking**: it spreads
all of `cite`'s key/value pairs into the new dict, then adds `"type"`. It's the
dict equivalent of "copy these fields and add one more."

Note also that `app.py`'s `events()` is *itself* a generator, and FastAPI's
`StreamingResponse` consumes it — generators compose naturally.

---

## 11. Decorators and caching

**Concept.** A **decorator** wraps a function to add behavior, using the `@name`
syntax above a `def`. You've already seen `@dataclass`. Here's one that adds
caching for free.

**Code** (`rag.py`):

```python
from functools import lru_cache


@lru_cache(maxsize=256)
def embed_query(question: str) -> list[float]:
    """Embed the question. input_type='query' is the search-side counterpart ..."""
    result = _voyage.embed(
        [question], model=VOYAGE_MODEL, input_type="query", output_dimension=EMBED_DIM
    )
    return result.embeddings[0]
```

**Why it works this way.** `lru_cache` (Least Recently Used cache) remembers the
return value for each set of arguments. The first time you call
`embed_query("What is RAG?")`, it makes the network call to Voyage and stores
the result. Every later call with that *same* question returns the stored vector
instantly — no API call, no latency, no cost. `maxsize=256` caps how many
distinct questions it remembers, evicting the least-recently-used when full.

This works precisely because embedding is **deterministic and side-effect-free**:
the same input always yields the same output, so a cached answer is always
correct. Don't cache functions whose results change over time or that have side
effects.

Decorators stack, too. In `app.py` an endpoint wears three:

```python
@app.post("/ask")
@limiter.limit(RATE_LIMIT)
def ask(request: Request, body: AskRequest) -> AskResponse:
    ...
```

`@app.post("/ask")` registers the route; `@limiter.limit(...)` adds rate
limiting. They apply bottom-up around the function.

---

## 12. Context managers (`with`)

**Concept.** A `with` block guarantees setup and cleanup happen as a pair — the
cleanup runs even if an exception is raised inside the block. The classic use is
resources that must be closed: files, network connections, database sessions.

**Code** (`embed.py`):

```python
with psycopg.connect(DB_URL) as conn:
    register_vector(conn)
    doc_ids = upsert_documents(conn, documents)
    store_chunks(conn, chunks, embeddings, doc_ids)
```

**Why it works this way.** `psycopg.connect(...)` opens a database connection.
The `with` ensures the connection is committed and closed when the block ends —
whether it ends normally or because something raised an error partway through.
You never have to remember a `finally: conn.close()`.

You can nest `with` blocks, and a single connection can be opened with options
(`rag.py`):

```python
with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
    register_vector(conn)
    hits = rerank_search(conn, question)
```

`row_factory=dict_row` makes query results come back as dicts (so you write
`row["title"]`) instead of plain tuples.

Opening a file for appending uses the same pattern (`evals/metrics.py`):

```python
with LOG_FILE.open("a") as f:
    f.write(json.dumps(record) + "\n")
```

The file is flushed and closed automatically at the end of the block.

The streaming example in §10 nests a `with` *inside* a generator — the SDK's
`client.messages.stream(...) as stream` keeps the HTTP stream open exactly as
long as the generator is producing values.

---

## 13. Exceptions and retries

**Concept.** `try`/`except` handles errors. You can catch *specific* exception
types, group several into a tuple, and re-raise when you've given up. A common
real-world use is retrying transient network failures with backoff.

**Defining which errors are worth retrying** (`embed.py`):

```python
_TRANSIENT_ERRORS = (
    voyageai.error.ServiceUnavailableError,
    voyageai.error.ServerError,
    voyageai.error.APIConnectionError,
    voyageai.error.Timeout,
    voyageai.error.RateLimitError,
)
```

**The retry loop** (`embed.py`):

```python
def _embed_with_retry(client, batch, attempts=5):
    for attempt in range(attempts):
        try:
            return client.embed(
                batch, model=VOYAGE_MODEL, input_type="document",
                output_dimension=EMBED_DIM,
            )
        except _TRANSIENT_ERRORS as exc:
            if attempt == attempts - 1:
                raise
            wait = 2**attempt
            print(f"  Voyage error ({type(exc).__name__}); retrying in {wait}s ...")
            time.sleep(wait)
```

**Why it works this way.**

- `except _TRANSIENT_ERRORS as exc` catches *any* of the exception types in that
  tuple, and binds the caught exception to `exc` so we can inspect it.
- Catching a **specific** set (not a bare `except:`) is deliberate. Auth errors
  and bad-request errors are *not* in the tuple, so they aren't retried —
  retrying them would just waste time, since they'll fail identically every time.
- `if attempt == attempts - 1: raise` — on the final attempt, give up and
  re-raise the original exception so the caller sees the real failure.
- `wait = 2**attempt` is **exponential backoff**: wait 1s, then 2s, 4s, 8s. This
  is gentler on an overloaded server than hammering it immediately.
- `return` inside `try` means a success exits the loop (and the function)
  immediately.

**Raising your own errors to fail fast** (`embed.py`):

```python
if not os.environ.get("VOYAGE_API_KEY"):
    raise SystemExit("VOYAGE_API_KEY is not set. Add it to .env or export it.")
```

`SystemExit` with a message prints it and exits with a non-zero status — a clean
way to abort a script with a helpful message instead of a stack trace.

---

## 14. Modules, imports, and environment config

**Concept.** Each `.py` file is a **module**. You pull names from other modules
with `import`. This repo's modules form the pipeline, and later stages import
earlier ones.

**Importing your own modules** (`embed.py`):

```python
from chunk import Chunk, chunk_corpus
from ingest import CORPUS_ROOT, Document, load_corpus
from rag import DB_URL  # local default, or DATABASE_URL if set
```

This is how `embed.py` reuses the `Document` dataclass and `load_corpus`
function defined in `ingest.py`, rather than redefining them. Notice the single
source of truth: `DB_URL` is defined once in `rag.py` and imported wherever it's
needed.

**Reading configuration from the environment** (`rag.py`):

```python
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
TOP_K = 5
```

**Why it works this way.**

- `load_dotenv()` reads a local `.env` file and loads its `KEY=value` lines into
  the environment — so secrets like `VOYAGE_API_KEY` live in a file you don't
  commit, not in the code.
- `os.environ.get("DATABASE_URL", "<default>")` reads an environment variable,
  falling back to a sensible default (a local Docker Postgres) when it isn't
  set. This is the idiom for "configurable, with a default": works out of the box
  locally, overridable in production by setting the variable.
- Module-level constants in `ALL_CAPS` (`TOP_K`, `EMBED_DIM`, `CLAUDE_MODEL`) are
  the Python convention for "this is a constant." `chunk.py` even mixes the two:
  `CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "256"))` — env-overridable, but
  with a default and converted from string to `int`.

---

## 15. Pydantic and FastAPI: validation at the edge

**Concept.** **Pydantic** models declare the shape of data and validate it
automatically. **FastAPI** uses them to validate incoming requests and serialize
outgoing responses. This is how you stop bad data at the boundary of your system.

**Defining request and response shapes** (`app.py`):

```python
from typing import Annotated
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]


class Source(BaseModel):
    title: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[Source]
```

**Wiring them into an endpoint** (`app.py`):

```python
@app.post("/ask")
@limiter.limit(RATE_LIMIT)
def ask(request: Request, body: AskRequest) -> AskResponse:
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        hits = search(conn, body.question)
    if _no_relevant_hits(hits):
        return AskResponse(answer=NO_ANSWER, citations=[], sources=[])
    text, citations = answer(body.question, hits)
    return AskResponse(
        answer=text,
        citations=[Citation(**c) for c in citations],
        sources=[
            Source(title=h["title"], source=h["source"], distance=h["distance"])
            for h in hits
        ],
    )
```

**Why it works this way.**

- Because the parameter is typed `body: AskRequest`, FastAPI automatically reads
  the JSON request body, validates it against `AskRequest`, and rejects anything
  malformed with a 422 error — *before* your function runs. An empty question or
  one over 1000 chars never reaches your code.
- `Annotated[str, Field(min_length=1, max_length=...)]` attaches validation rules
  to the `str` type: the question must be a non-empty string within a length
  bound. This is the cost-control lever — the one input a caller controls — being
  bounded declaratively.
- The `-> AskResponse` return type means FastAPI serializes your returned object
  to JSON, guaranteeing the response matches the documented schema.
- `Citation(**c)` constructs a `Citation` model from a dict `c` by unpacking its
  keys as keyword arguments — the same `**` unpacking from §10, here building an
  object instead of a dict.

This "validate at the edge" pattern means the *interior* of the system can trust
its data. By the time `search` runs, `body.question` is guaranteed to be a
reasonable string.

---

## 16. Putting it together: the full pipeline

Now that you've seen the pieces, here's how they compose into a system. Each
stage is a module you can run on its own, and each imports the one before it.

```
ingest.py  →  chunk.py  →  embed.py  →  rag.py  →  app.py
  load        split        vectorize    retrieve   serve
  corpus      into         + store      + ask       over
  to text     chunks       in pgvector  Claude      HTTP
```

**1. Ingest** (`ingest.py`) turns a folder of mixed files into clean
`Document`s, dispatching on extension (§8) and using `pathlib` (§7):

```python
documents = load_corpus(CORPUS_ROOT)
```

**2. Chunk** (`chunk.py`) splits each document into overlapping, token-sized
`Chunk`s using a closure with `nonlocal` (§9):

```python
chunks = chunk_corpus(documents)
```

**3. Embed** (`embed.py`) turns chunk text into vectors (with retries, §13) and
stores them in Postgres inside a `with` block (§12):

```python
embeddings = embed_batches(client, texts, tracker)
with psycopg.connect(DB_URL) as conn:
    register_vector(conn)
    doc_ids = upsert_documents(conn, documents)
    store_chunks(conn, chunks, embeddings, doc_ids)
```

**4. Retrieve + answer** (`rag.py`) embeds the question (cached, §11), finds the
nearest chunks, and asks Claude over them, streaming the result (§10):

```python
with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
    register_vector(conn)
    hits = rerank_search(conn, question)
text, citations = answer(question, hits)
```

**5. Serve** (`app.py`) wraps all of it in a validated HTTP API (§15):

```python
@app.post("/ask")
def ask(request: Request, body: AskRequest) -> AskResponse:
    ...
```

### What to do next

- **Read the real files** in the order above. You now have the vocabulary to
  follow every line. The comments in those files explain the *domain* decisions
  (why 256-token chunks, why hybrid search) — this tutorial covered the *Python*.
- **Run a stage** and read its output: `uv run ingest.py` prints corpus stats,
  `uv run chunk.py` prints sample chunks.
- **Trace one query** end to end: start in `app.py`'s `ask`, follow it into
  `rag.py`'s `search` → `embed_query` → the SQL, then `answer` → Claude.
- **Make a small change** and see what happens: adjust `TOP_K` in `rag.py`, or
  add a new file extension to `EXTRACTORS` in `ingest.py`.

The best way to learn Python is to modify code that already works. This
repository is a good place to do exactly that.
