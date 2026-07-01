# Eval Design — innerdance RAG

_Date: 2026-06-16 · Source: Hamel Husain, ["Your AI Product Needs Evals"](https://hamel.dev/blog/posts/evals/) (first full read). This maps his three-level framework onto our system in our own terms — what we already have, what's missing, and the order to build it._

## Two principles that come before any metric

1. **Error analysis before metrics.** Don't pick numbers first and hope they mean something. Look at real traces, name the failure modes, *then* write the assertion or metric that catches each one. We already did one round of this in `failure-analysis.md` — that document, not a dashboard, is the model for how Level 2 should work.
2. **If everything passes, the eval is too easy.** Hamel: "one signal you are writing good tests is when the model struggles to pass them." We are sitting in exactly the failure this warns about — 25 questions, every answer graded good, recall@5 at ceiling. That's not proof the system is great; it's proof the eval can't yet see the next tier of failures. So the highest-value work is making the eval *harder*, not making the RAG better.

These two set the priority order at the bottom. The three levels are the structure; the principles decide what to build first.

## Where we stand today

We already have pieces of Levels 1 and 2. Level 3 doesn't exist yet because we have no real users. The table below is the whole plan in one glance; the sections expand it.

| Level | What it is here | Status |
|---|---|---|
| **1 — Assertions** | Cheap, deterministic, per-commit checks on a single response | Partial — checks exist in code, not as tests |
| **2 — Model & human eval** | Retrieval metrics, human grading, LLM judge over logged traces | Retrieval + human grading done; **LLM judge missing** |
| **3 — Production / A/B** | Real-user feedback and online metrics, live retriever A/B | Not started (deferred until there are users) |

## Level 1 — Assertions (deterministic, every commit)

Fast pass/fail checks that need no model call and no human. Hamel's point is that these double as guardrails (data cleaning, retries), not just tests — several of ours already run *inside* the request path. The job now is to pull the ad-hoc ones into a real `pytest` file so CI runs them.

| Assertion | What it catches | Where it lives now |
|---|---|---|
| Every citation's `cited_text` is a substring of its chunk | Fabricated quotes (the citation isn't real) | Informal check in `rag.py __main__` → promote to a test |
| `chunk_id` of each citation maps back to a retrieved hit | Citation pointing at nothing | Guaranteed by construction in `_citations` → assert it |
| No-answer gate fires when top-1 distance > `RELEVANCE_THRESHOLD` | Answering off-topic questions from nothing | `_no_relevant_hits` in `app.py` → test both sides of the gate |
| Same question → same retrieved IDs | Non-determinism that would make grading meaningless | Relied on everywhere → pin it with a test |
| Over-long / empty questions are rejected at the boundary | Unbounded embed+generate cost | Pydantic `Field(min_length=1, max_length=1000)` → assert the 422 |
| Answer is non-empty and contains ≥1 citation when hits exist | Silent generation failure | Not checked → add |

These are the cheapest signal we have and the only one that can run on every commit, so they go first mechanically even though they're not where the risk is.

## Level 2 — Model & human eval (the core loop)

This is where the real work lives. Hamel's prerequisite — "log your traces" — means every eval run records (question, retrieved chunks, answer, citations) so a human or a judge can inspect it. `eval_set.jsonl` is that trace store for the offline set. For the *live* path, `app.py` now traces each `/ask` to **Langfuse** (one trace per request: retrieved chunks as metadata, the Claude call captured automatically with model + token usage) — the production-side trace store, active whenever the `LANGFUSE_*` keys are set and a no-op otherwise.

**2a. Retrieval metrics — done.** `metrics.py` computes recall@5 and MRR against keyword-derived gold, logs each run to `metrics_log.jsonl`, and separates "tight gold" from inflated broad-keyword gold. This is the number we watch when changing chunking/retrieval (see `chunking-experiments.md`). Keep it.

**2b. Human grading — done.** `gen_eval.py` produces draft answers (explicitly *not* ground truth — they come from the system under test), and `grade.html` is the low-friction viewer Hamel insists on: question, RAG answer, ideal answer, and retrieved chunks side by side, with correct/partial/wrong/hallucinated labels persisted in the browser. This is our error-analysis surface.

**2c. LLM-as-judge — the missing piece.** Human grading doesn't scale to a growing eval set or to every CI run. The judge replaces the human for routine runs, but only after it's *aligned*:

- **Build the judge** to score one answer on a small rubric — groundedness (is every claim supported by the retrieved chunks?), correctness vs. the ideal answer, and appropriate refusal on no-answer questions. Use a model *stronger* than production for judging (production generates with Sonnet; the judge should use a top-tier model, e.g. Claude Opus) — the judge needs to be sharper than the thing it grades.
- **Align it against our human labels.** The grades already collected in `grade.html` are the alignment set. Iterate the judge prompt until judge↔human agreement is high, and measure it with **precision/recall, not raw agreement** — our label distribution is lopsided (almost everything is "correct"), so raw agreement is misleading. Hamel: "you must maintain a mini-evaluation system to track its quality." The judge is itself a thing we evaluate.
- **Re-check alignment** whenever the judge prompt or the judged model changes. Don't assume it stays aligned.

**Error-analysis cadence.** `failure-analysis.md` is the template: after a meaningful change, read the traces, count failures by category (retrieval miss / retrieval noise / generation failure / hallucination), and write down the honest caveat. The judge feeds this by flagging candidates; the human still reads the flagged ones.

## Level 3 — Production / A/B (deferred until real users)

Hamel reserves this for mature products with live traffic, which we are not. Recording the plan so we don't reach for it prematurely:

- **User feedback signal** — thumbs up/down on each answer in the frontend, plus citation click-through (we already reconstruct source passages via `/source/{chunk_id}`; click-through is a free relevance signal).
- **Online metrics** — refusal/no-answer rate, citation-click rate, latency, cost per query.
- **Live A/B between retrievers** — we have three (`search`, `hybrid_search`, `rerank_search`) and offline numbers that disagree (hybrid wins recall, rerank wins MRR). Offline can't break the tie; real-user A/B can. Until then, `metrics_log.jsonl` is the *offline analog* of this level — not a substitute for it.

## Priority order (set by the principles, not the level numbers)

1. **Harden the eval set first.** Tighten the loose gold (Q8/9/11 have 35–140 "gold" chunks) and grow the question set with harder, adversarial cases until the model *stops* passing everything. Per principle 2, this is the only way the rest of the machinery can detect anything. _Highest value, and it's not code — it's data and labeling._
2. **Lift Level-1 assertions into a `pytest` file** so CI runs them on every commit. _Done — `test_assertions.py` covers the pure, deterministic checks (citation groundedness, the no-answer gate, question-length bounds, the retrieval metrics), and `.github/workflows/ci.yml` runs `uv run pytest`. The DB/API-dependent checks (retrieval determinism, end-to-end answer) are integration tests and stay out of CI by design._
3. **Build and align the LLM judge (2c).** Only worthwhile once (1) gives it harder cases to discriminate and human labels to align against.
4. **Level 3** stays parked until there are real users.

## Structure — where the code lives

Everything eval-related lives under `evals/`:

```
evals/
  __init__.py            makes evals a package, so scripts run as `-m evals.x`
  DESIGN.md              this file
  eval_set.jsonl         the test bank (questions, ideal answers, gold labels)
  test_assertions.py     Level-1 pytest (run with `uv run pytest`)
  metrics.py             retrieval metrics (recall@5, MRR)
  metrics_log.jsonl      run-over-run log
  compare_retrieval.py   vector-vs-keyword diagnostic on hard queries
  gen_eval.py            draft-answer generator (input to grading)
  grade.py / grade.html  human grading harness
  judge.py               LLM-as-judge (to build, step 3)
  failure-analysis.md    error-analysis notes
  chunking-experiments.md  retrieval-change log
```

The app/pipeline modules live in the `rag/` package (`rag/query/retrieve.py`, `rag/query/answer.py`, `rag/app.py`, `rag/indexing/{ingest,chunk,embed}.py`); `rag/__init__.py` re-exports the public API, so the eval scripts still `import rag` unchanged. Run everything as modules from the repo root (e.g. `uv run python -m evals.metrics`), and each script anchors its data files with `Path(__file__).parent` so cwd doesn't matter. CI is `.github/workflows/ci.yml`.

### Extraction to `ai-utils` at month's end (target 2026-06-30)

Once the judge and metrics settle, the **corpus-agnostic** pieces graduate to the shared `ai-utils` package so the next RAG can reuse them:

- **Judge runner** — the generic loop: rubric prompt → model call → parse → structured score. The *rubric* stays here; the *runner* leaves.
- **Stats helpers** — `recall_at_k`, `reciprocal_rank`, judge↔human agreement (precision/recall), and a bootstrap CI helper for run-to-run significance.
- **Schemas** — the eval-row and judge-result Pydantic models.

**Stays project-specific (does not extract):** `eval_set.jsonl`, the innerdance judge rubric, and the keyword-based gold labeling — all tied to this corpus.

## Honest caveat

This plan is only as good as principle 1. If we build the judge and the assertions on top of the *current* easy eval set, we'll get a green board that means nothing — the same trap `failure-analysis.md` already named. Step 1 (harder questions, tighter gold) is load-bearing for everything below it.
