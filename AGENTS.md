# AGENTS.md

Guidance for AI agents (and humans) working on this repo. These are the current
defaults, not law: any convention or architecture here can change when a better
option is found — measure with the eval harness, then change it. What should not
happen silently is *accidental* drift from these choices.

## What this is

A RAG system over the innerdance corpus. Design rationale lives in `README.md` —
read it before restructuring anything.

- `rag/` — backend: `indexing/` (ingest → chunk → embed), `query/` (retrieve →
  answer, plus two agentic paths), `app.py` (FastAPI), `db.py` (connection, `Hit`
  row shape, `EMBED_DIM`), `config.py` + `config.toml` (hyperparameters)
- `evals/` — the quality gate (see Testing below); `evals/DESIGN.md` explains the levels
- `db/migrations/` — ordered SQL migrations
- `frontend/` — Next.js UI; has its own `AGENTS.md`

## Quick commands

```
uv run pytest                              # unit assertions (fast, no DB/network)
uv run ruff check . && uv run ruff format .  # lint + format (CI enforces both)
uv run fastapi dev rag/app.py              # serve the API on :8000
uv run python -m db.migrate                # apply pending schema migrations
uv run python -m rag.pipeline              # rebuild the index (costs Voyage credits)
uv run python -m evals.run --config evals/configs/baseline.json  # full eval (costs LLM credits)
```

## Conventions

- **Smallest correct change.** No speculative abstractions, no scaffolding for
  later, no new dependency for what a few lines can do. Deliberate shortcuts are
  marked with a `ponytail:` comment naming the ceiling and upgrade path — keep those.
- **Comments sparingly.** Don't restate code, config keys, or narrate change
  history; don't point at the README from inline comments (one pointer per module
  docstring is enough). DO comment the genuinely unintuitive: non-obvious
  invariants, workarounds for external bugs, deliberate trade-offs.
- **No import-time side effects.** Clients, DB connections, and compiled agents
  are built lazily in cached factories (`rag/clients.py`, `deepagent._agent()`).
  `import rag.app` must succeed with no API keys, no `.env`, no Postgres — CI
  relies on this.
- **Env access:** `os.environ[key]` for required secrets (fail loud at first use);
  `.get(key, default)` only where a default is legitimate (`DATABASE_URL`,
  `FRONTEND_ORIGIN`). Document new vars in `.env.example`.
- **`config.toml` is for tunable hyperparameters** (retrieval/generation knobs an
  eval config might sweep). App policy (rate limits, size caps) and
  schema-coupled constants (`EMBED_DIM`) stay in code, next to what they bind to.
- **Typed payloads:** retrieval hits are `Hit`, citations are `Citation`
  (TypedDicts). Extend those rather than reintroducing bare `list[dict]` for the
  same shapes.

## Testing policy (deliberate — don't "fix" it)

`evals/test_assertions.py` holds pure, deterministic unit assertions only (no DB,
no network) and runs in CI. End-to-end quality is gated by the eval harness
(`evals/run.py` + `check_regression.py`), not by mocked integration tests — do not
add coverage targets or mock-heavy suites. If you touch retrieval, generation, or
prompts, run the eval and compare against the previous run before merging.

## Gotchas

- `EMBED_DIM` (in `rag/db.py`) must match the `VECTOR(1024)` columns in
  `db/migrations/` — changing it means a migration plus a full re-embed.
- Both web-search agents are kept deliberately: `web_search_graph_agent.py`
  serves `/agent/stream`; `web_search_agent.py` is the eval baseline
  (`evals/web_search/run_baseline.py --impl loop`) and home of the shared web
  tooling. Don't delete either without an eval verdict.
- `db.migrate` and the app both resolve `DATABASE_URL` through `.env` — with a
  production URL there, migrations run against production.
- `agent_runs` in `rag/app.py` is an in-memory, single-process store; SSE replay
  breaks across restarts/workers (marked `ponytail:`).
- The first `/ask/agent` request pays the one-time deep-agent build
  (checkpointer setup + compile).
