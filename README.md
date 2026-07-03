# rag-system

Retrieval-augmented Q&A over the **innerdance** corpus — a set of innerdance-related
documents (talk transcripts, a book, dialogue and slide PDFs). The goal is to query
innerdance concepts easily and surface supporting material that connects topics, with
every answer grounded in cited source passages.

## Layout

```
rag/
  db.py              shared DB_URL + connect() helper
  indexing/          build the index — run once, in order
    ingest.py        1. load the corpus into clean Documents
    chunk.py         2. split Documents into overlapping, metadata-rich chunks
    embed.py         3. embed chunks with Voyage and store them in pgvector
  query/             per request
    retrieve.py      hybrid search (vector + keyword, RRF) then rerank
    answer.py        generate the answer with grounded citations
  app.py             FastAPI service
  ocr.py             one-off: OCR image-only slide PDFs to Markdown
  pipeline.py        build_index(): runs ingest -> chunk -> embed in sequence
db/migrations/       ordered schema migrations; db/migrate.py applies pending ones
evals/               evaluation harness and research artifacts
frontend/            Next.js UI
```

## Setup & running

```bash
docker compose up -d                     # Postgres + pgvector (empty)
cp .env.example .env                      # then fill in the API keys
uv sync
uv run python db/migrate.py               # apply pending schema migrations

uv run python -m rag.ocr                  # optional: OCR the image-only slide PDFs first
uv run python -m rag.pipeline             # build the index (ingest -> chunk -> embed -> store)
uv run fastapi dev rag/app.py             # serve the API on :8000
```

Inspect a stage without embedding: `uv run python -m rag.indexing.ingest` (corpus shape),
`uv run python -m rag.indexing.chunk` (chunk stats + samples).

Env vars (`.env`): `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, optional `OPENROUTER_API_KEY`
(alternate provider), `DATABASE_URL` (defaults to the local docker container),
`FRONTEND_ORIGIN`, and the `LANGFUSE_*` keys for tracing.

## Pipeline order (build the index)

`rag/indexing/` runs once, in order, to populate the database:

1. **ingest** — load each source file (RTF, PDF, EPUB, HTML, Markdown) into a `Document`
   with clean text and metadata. The Maia transcripts arrive as one timecoded ASR
   fragment per line; the timecodes are ~40% of the tokens and fragment every sentence,
   so we strip them and reflow into sentence-split prose (timing is kept as chunk
   metadata). The dialogue ebook (`transformation_medicine_ebook.pdf`) is a two-voice
   dialogue whose speakers are encoded only by font (Pi in italic, Doc Romy in roman), so
   it's parsed per-span to recover who's speaking.
2. **chunk** — split each document into ~256-token chunks with ~50-token overlap. The
   splitting unit is the whole **line** (a timestamped utterance for transcripts, a
   paragraph for the book), so boundaries never fall mid-word. Each chunk carries the
   document metadata plus its nearest preceding heading.
3. **embed** — embed each chunk with Voyage (`voyage-4`, 1024-dim) and store the vector in
   pgvector. Before embedding, each chunk's text is prefixed with `section — title`; this
   **contextual prefix** measurably improves retrieval (recall 0.89 → 1.00 in the chunking
   experiments). Only the vector sees the prefix — the stored content stays raw. Storing
   is idempotent: a document's chunks are replaced, not duplicated, on re-run.

Two adopted retrieval tunings (see `evals/chunking-experiments.md`): the contextual prefix
above, and smaller chunks (512 → 256 tokens), which lifted MRR 0.72 → 0.85.

## Retrieval design (per query)

Three stages, in `rag/query/retrieve.py`:

1. **Vector search** — cosine nearest neighbours over the chunk embeddings. Catches
   paraphrase, but being embedding-based it blurs exact rare terms.
2. **Keyword search** — Postgres full-text (`ts_rank`). `websearch_to_tsquery` ANDs every
   term, which is too strict for a full question (one missing word matches nothing), so we
   swap `&` for `|` to OR the terms. Nails exact rare terms vectors blur, but being
   exact-lexeme it can't see past typos.
3. **Hybrid fusion (RRF)** — the two rankings are combined with weighted Reciprocal Rank
   Fusion: each chunk scores `sum(weight / (RRF_K + rank))`. Fusing on *rank*, not score,
   is what lets it blend cosine distance and `ts_rank` (two incomparable scales) without
   normalizing. `RRF_K = 60` (the original RRF paper's value) damps how much exact rank
   matters. Keyword is down-weighted (`0.5` vs vector `1.0`) as the noisier signal — that
   weight won a sweep: recall@5 **0.79** vs **0.74** for both pure-vector and equal-weight
   fusion (`evals/metrics_log.jsonl`).
4. **Rerank** — a cross-encoder (`rerank-2.5`) re-scores the top 20 hybrid candidates,
   reading each `(question, chunk)` pair together — a sharper signal than the independent
   first-stage scores, but too slow to run over the whole corpus.

**Relevance gate.** If even the nearest chunk is farther than `RELEVANCE_THRESHOLD = 0.7`
(cosine distance), the corpus almost certainly doesn't cover the question, so the API
refuses ("I don't have information on that") instead of generating — avoiding hallucination
and the generation cost. Grounded in observed top-1 distances: on-topic ~0.40, a recoverable
typo'd query 0.63, a genuine no-answer 0.94 (`evals/failure-analysis.md`). Retune if the
corpus or embedding model changes.

## Generation / provider seam

`rag/query/answer.py` answers over the retrieved chunks behind one `answer()` that
dispatches by `provider` (default `GEN_PROVIDER = "anthropic"`, so production is unchanged):

- **anthropic** — the native **Citations API**. Chunks are passed as separate document
  blocks, so each returned citation's `cited_text` is a quote the API extracts from the
  source: it can't fabricate a quote, and `document_index` maps back to the retrieved hit.
- **openai-compat** — an OpenAI-compatible endpoint (OpenRouter, via `instructor`) for a
  cost-efficient alternative model. There's no Citations API off-Anthropic, so instead of
  having the model re-transcribe verbatim quotes (token-heavy and error-prone on weaker
  models) we ask only *which* numbered chunks support each claim. Each citation's
  `cited_text` is then the chunk itself — grounded by construction. The trade-off is
  granularity: a whole chunk, not the exact supporting sentence. Uses `Mode.JSON` because
  some OpenRouter models don't reliably emit tool calls for the schema, and retries on a
  length cutoff (OpenRouter occasionally truncates under load).

Per-query cost is bounded on every axis: the question length is capped at the API boundary,
retrieval sends a fixed `TOP_K` chunks, and `MAX_TOKENS` caps the output.

## API

`rag/app.py` (FastAPI):

- `POST /ask` — `{"question": "..."}` → `{answer, citations, sources}`.
- `POST /ask/stream` — Server-Sent Events: `text` events stream the answer, then one
  `citation` event per source.
- `GET /source/{chunk_id}` — the cited chunk reconstructed in place (`before`/`chunk`/
  `after`) for click-through highlighting in the frontend.

Requests are rate-limited per client IP (`10/minute`, in-memory → per-process). CORS allows
the frontend origin (`FRONTEND_ORIGIN`). With the `LANGFUSE_*` keys set, each request is one
Langfuse trace, the generation call auto-captured by the active provider's instrumentor
(`AnthropicInstrumentor`, or the `langfuse.openai` drop-in for the OpenAI-compatible path).

## Evals

`evals/` holds the evaluation harness (retrieval metrics, an LLM-as-judge, regression gate)
and its research artifacts. See `evals/DESIGN.md`.
