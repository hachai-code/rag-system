# rag-system Improvement Plan

## Context

Compared `~/Documents/Code/rag-system` against `~/PycharmProjects/pydantic-ai` and `~/PycharmProjects/software-agent-sdk` (OpenHands). Both reference repos are large production frameworks; the goal is to borrow only the cleanliness conventions that pay off for a ~3,000-line single-developer project, and to fix the concrete issues found. rag-system's evals-as-quality-gate design (documented in `evals/DESIGN.md`) is deliberate and stays.

**Where the reference repos are cleaner:**
1. **Enforced tooling** — both configure ruff in pyproject + pyright + pre-commit + CI lint gates. rag-system has zero lint/format/type config; CI only runs pytest.
2. **No import-time side effects** — pydantic-ai builds SDK clients at provider construction with env fallbacks. rag-system builds a Voyage client at import (`rag/query/retrieve.py:47`) and opens 2 live Postgres connections + compiles the deep agent at import (`rag/query/deepagent.py:251-357`) — CI carries dummy env vars just to import the package.
3. **Typed payloads** — pydantic-ai uses dataclasses/TypedDicts everywhere; rag-system passes `list[dict]` for retrieval hits despite a stable shape.
4. **Public/private discipline** — pydantic-ai gates surface with `__all__`, underscore modules. rag-system re-exports privates from `rag/__init__.py` and has cross-module private imports (`deepagent.py:42-50` imports `_voyage`, `_distill`, `_encoder`...).
5. **Single source of truth** — sdk enforces layered imports via a pre-commit AST script. rag-system duplicates `DB_URL`, `EMBED_DIM`, retrieval-gate logic, pass-rate aggregation, and carries two parallel web-search agents (a known, intentional A/B).

**Corrections found during verification:** the `@lru_cache` on `embed_query`/`multi_query` is bounded (256) and deliberate — keep. Working-tree cruft is mostly already gitignored; only 2 files are actually tracked.

Each phase below is independently shippable. Effort: S = <30 min, M = 1-3 h.

---

## Phase 0 — Housekeeping (S)

- Copy this plan into the repo as `rag-system/IMPROVEMENT_PLAN.md` (source: `~/.claude/plans/compare-rag-system-with-pydantic-ai-mighty-cascade.md`) so it's accessible alongside the code; delete it when all phases are done.

- `git rm --cached .deepeval/.deepeval_telemetry.txt .humanlayer/workspace.json`; add `.deepeval/` to `.gitignore`, widen `.humanlayer/tasks/` to `.humanlayer/`.
- Add missing `ANTHROPIC_API_KEY=` and `DATABASE_URL=` (with default noted) to `.env.example`.

## Phase 1 — Ruff + CI enforcement (S; do first so later diffs stay clean)

1. `pyproject.toml`: add `ruff` to dev group, plus:
   ```toml
   [tool.ruff]
   line-length = 100
   [tool.ruff.lint]
   select = ["E", "F", "I", "UP", "B"]
   ```
   (pydantic-ai's core selection + bugbear; 100 cols matches existing code width.)
2. One isolated mechanical commit: `uv run ruff check --fix . && uv run ruff format .`
3. Add to `.github/workflows/ci.yml` before pytest: `uv run ruff check . && uv run ruff format --check .`
   Deliberately **no pre-commit** — CI + editor covers one developer.

## Phase 2 — Import hygiene (unlocks removing CI dummy env vars)

- **2.1 (S)** Lazy Voyage client: move to `rag/clients.py` as `@lru_cache(maxsize=1) def voyage_client()`; update `retrieve.py` call sites and `deepagent.py:43` (which imports private `_voyage`).
- **2.2 (M)** Lazy deep-agent build: wrap PostgresSaver/PostgresStore setup + `create_deep_agent(...)` (`deepagent.py:357`) in one cached `_agent()` factory called by `run_/stream_/resume_deepagent`. Leave the thread-id dicts.
- **2.3 (S)** Slim `rag/__init__.py`: drop private re-exports (`_chunk_citations`, `_citations`) and tuning constants from `__all__`; keep the public call surface (`search`, `retrieve`, `answer`, `answer_stream`, `Claim`, `GroundedAnswer`, ...). Keep `load_dotenv()`. Fix callers/tests to import from defining modules.
- **2.4 (S)** Delete the dummy `VOYAGE_API_KEY`/`ANTHROPIC_API_KEY` block from `ci.yml` — the proof 2.1/2.2 worked.

## Phase 3 — Kill known duplication (each ships alone)

- **3a (S)** `db/migrate.py:18`: `from rag.db import DB_URL` (after Phase 2 makes the import cheap).
- **3b (S)** `EMBED_DIM`: define once in `rag/db.py` (it's schema-coupled to `VECTOR(1024)`, not tunable — not config.toml); import in `retrieve.py` and `indexing/embed.py`.
- **3c (S)** One retrieval-gate helper in `retrieve.py` — `covered(conn, question, threshold) -> tuple[bool, list[dict]]` — used by `app.py:154`, `app.py:193`, `evals/run.py:85`. Keeps prod and eval gates identical by construction.
- **3d (S)** Extract `_pass_rate(rows)` into `evals/schema.py` (already the shared import-light module); use from `evals/api.py` and `evals/run.py`.
- **3e (M, biggest win)** Web-search A/B: do NOT refactor both — run the existing `evals/web_search/` eval, pick the winner, delete the loser (−330 to −430 lines). If the graph agent wins: first move shared helpers (`_distill`, `_encoder`, `_cited_urls`, `fetch_page`, `search_web`, budgets) into `rag/query/web_tools.py` with public names (also fixes most cross-module private imports), then delete `web_search_agent.py`.

## Phase 4 — Type the hit payload (S/M)

- One `TypedDict` in `retrieve.py`:
  ```python
  class Hit(TypedDict):
      id: int; title: str; source: str; content: str; distance: float
  ```
  Annotate retrieval functions `-> list[Hit]` and `answer(question, hits: list[Hit], ...)`. Zero runtime change. Citations too if equally stable; skip agent-run dicts (still evolving).

## Phase 5 — Comment thinning (S/M, per-file as you touch them)

The comments are good "why" comments but too dense — rationale is duplicated between code and README, and pointer comments clutter constant blocks (`retrieve.py:17-45` has "see README" three times in one screen; every `AskRequest` field in `app.py:80-95` carries an inline comment). Reference-repo rule (sdk AGENTS.md): comments sparingly, never restating, rationale lives in docs.

Rules for the pass:
- Delete comments that only point to README — keep ONE pointer in the module docstring (`retrieve.py` already has it, line 3-4).
- Delete comments that restate the config key or the default value next to `CONFIG.x` reads.
- Pydantic request fields: move genuinely useful field comments into `Field(description=...)` so they surface in the OpenAPI `/docs` instead of only in source; delete the rest.
- KEEP non-obvious why-comments (the tsquery `&`→`|` trick, query-vs-document embedding, fail-open rationale, all `ponytail:` markers).
- Docstrings: contract stays, design-history essays that duplicate README move there or die.

Do it opportunistically per file when a phase above already touches it (retrieve.py in 2.1/3c/4, app.py in 3c, deepagent.py in 2.2) — not as a repo-wide sweep commit.

## Phase 6 — AGENTS.md (S, new file at repo root)

One short file (~60-80 lines, not sdk's 422) borrowing the structure of software-agent-sdk's AGENTS.md and the distilled-ruleset idea of pydantic-ai's `agent_docs/index.md`, scoped to what an agent working on this repo actually needs:

- **Project map** (5 lines): `rag/` pipeline stages (ingest → chunk → embed → retrieve → answer), `evals/` as the quality gate, `db/` migrations, `frontend/`. Point to README for design rationale — AGENTS.md does not duplicate it.
- **Quick commands**: `uv run pytest`, `uv run fastapi dev rag/app.py`, `uv run python -m evals.run`, migration command, ruff commands (once Phase 1 lands).
- **Code conventions** (adapted from sdk `<CODE>` section):
  - Comments sparingly; never restate code, config keys, or point to README from constant blocks (one pointer in the module docstring suffices); DO comment genuinely unintuitive things (invariants, external-bug workarounds, deliberate trade-offs). Keep `ponytail:` markers.
  - No import-time side effects (clients/DB connections built lazily — post Phase 2).
  - `os.environ[key]` for required secrets, `.get()` only with a legitimate default.
  - `config.toml` = retrieval/generation hyperparameters only; app policy constants stay in code.
  - No speculative abstractions; smallest correct diff.
- **Testing policy** (from DESIGN.md, so agents stop proposing coverage): pure unit assertions in `evals/test_assertions.py`; the eval harness is the real gate — run it when touching the answer path; no mocked-integration tests.
- **Gotchas**: `EMBED_DIM` is coupled to `VECTOR(1024)` in migrations; web-search agent A/B in flight (don't refactor either, see Phase 3e); `agent_runs` store is in-memory/single-process.

Written last (after Phases 1-5) so it documents the post-cleanup state rather than aspirations.

## Phase 7 — Config drip (opportunistic)

- Rule: move a hardcoded knob to `config.toml` the first time you actually retune it (`N_VARIANTS`, agent budgets — but wait for 3e). Leave `MAX_QUESTION_CHARS`/`RATE_LIMIT` hardcoded (app policy, not hyperparameters). Don't restructure the flat `Config` dataclass.
- Normalize env access: `os.environ[key]` for required secrets, `.get(key, default)` only where a default is legitimate.

## Phase 8 — Create a reusable clean-code refactoring Skill (final step, M)

Distill this comparison into a personal skill at `~/.claude/skills/cleaning-up-codebases/SKILL.md` so the playbook applies to any future codebase. Technique-type skill, <500 words, following superpowers:writing-skills conventions (test-first, description = trigger conditions only).

**Frontmatter:**
```yaml
---
name: cleaning-up-codebases
description: Use when asked to clean up, modernize, or improve the code quality of an existing codebase, or when a project has no lint/type tooling, import-time side effects, duplicated constants, or untyped dict payloads
---
```

**Body — the playbook proven here, generalized:**
1. **Scale rule first** (the core principle): match rigor to project size — solo project ≠ framework. Explicitly list what NOT to cargo-cult (coverage targets, strict type gates, pre-commit, ABC hierarchies, exceptions modules without library consumers).
2. **Ordered phases, each independently shippable:** (a) housekeeping (gitignore, .env.example completeness); (b) lint/format config + CI gate BEFORE any code edits, one isolated autofix commit; (c) import hygiene — no clients/connections/compiled artifacts at module import; cached factories instead (acceptance test: `import pkg` succeeds with no secrets and no services running); (d) kill duplication — single source for constants/URLs/shared logic, and for A/B experiments "decide the eval, delete the loser", never refactor both; (e) type the payload currency (TypedDict for stable dict shapes, zero runtime change); (f) comment thinning (sdk policy: sparingly, never restate or point elsewhere without a sync mechanism, DO keep genuinely unintuitive why-comments); (g) AGENTS.md last, documenting the post-cleanup state.
3. **Verification contract:** existing tests after every phase; the import-purity check; the project's real quality gate (evals/e2e) after behavior-touching phases.
4. **One example:** the lazy-client factory before/after (module-level `client = SDK()` → `@lru_cache def client()`).

**Scope note:** complements existing skills rather than duplicating them — `refactor-architecture` (structural moves), `ponytail-audit` (over-engineering deletion); this one is the bring-up-hygiene playbook.

**Test per writing-skills TDD (required before deploying):**
- RED: point a fresh subagent at a small messy repo (or rag-system pre-cleanup) with "clean this up" and no skill — document what it does (typical baseline failure: repo-wide sweep in one commit, cargo-culted strict tooling, refactors both sides of an A/B).
- GREEN: same scenario with the skill — verify it phases the work, adds proportionate tooling, and asks the eval to decide the A/B.
- REFACTOR: counter any new failure modes found, re-test.

---

## Explicitly NOT doing (framework practices that don't pay off at this scale)

1. pyright/mypy CI gate (run `uvx pyright rag/` ad hoc if curious)
2. pre-commit framework
3. Custom exceptions module (no library consumers)
4. logging-module sweep of the 69 `print()`s (mostly correct CLI output)
5. Dataclass/ABC/discriminated-union machinery for retrieval
6. Tests-mirror-package-layout / coverage targets (DESIGN.md's evals-as-gate stands)
7. Touching the bounded `lru_cache` on `embed_query`/`multi_query`
8. "Fixing" `app.py:41` importing `evals.api` (ships together in the only deployment; `try/except ImportError` if it ever bites)

---

## Verification

| Check | Command | When |
|---|---|---|
| Unit tests | `uv run pytest` | every phase |
| Lint/format | `uv run ruff check . && uv run ruff format --check .` | Phase 1 onward |
| Import purity | `import rag, rag.app` in a shell with no API keys and no Postgres — must succeed | Phase 2 acceptance test |
| App boot | `uv run fastapi dev rag/app.py` serves `/docs` | Phases 2-3 |
| Eval harness (real gate) | `uv run python -m evals.run` + regression check | after 2.2, 3c, and 3e (where it IS the decision) |

Behavior-neutral phases (0, 1, 3a/3b/3d, 4, 5, 6, 7): pytest + ruff suffice. 2.2, 3c, 3e touch the answer path — run the eval harness before merging.

## Critical files

- `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`, `.env.example`, `AGENTS.md` (new)
- `rag/__init__.py`, `rag/clients.py`, `rag/db.py`
- `rag/query/retrieve.py`, `rag/query/deepagent.py`, `rag/query/answer.py`
- `rag/query/web_search_agent.py`, `rag/query/web_search_graph_agent.py`
- `rag/app.py`, `db/migrate.py`, `evals/run.py`, `evals/api.py`, `evals/schema.py`
