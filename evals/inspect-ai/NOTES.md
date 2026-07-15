# Inspect AI port — notes (2026-07-15)

Port of a stratified 15/75 subset of `evals/answer/data/rag_system_human_eval.jsonl`
to Inspect AI (`task.py`). Same rubrics (`RUBRICS`/`SYSTEM` imported from
`evals/answer/judge.py`), same pipeline (`covered → retrieve → answer`, flash, baseline
config), judge = `openrouter/deepseek/deepseek-v4-flash`. The hand-rolled harness
(`evals/run.py`) stays canonical; this exists to see what a framework buys.

Reference run: 15 samples in **1:20 wall-clock** at `--max-connections 5`
(run.py is a sequential loop — same work takes several minutes). Judge-side cost:
10.3k tokens. Per-code means: A 1.00, B 1.00, C 0.83, D 0.25, E 0.00, F 1.00, trace 0.67
(D/E weakness consistent with the taxonomy-v2 hardening — the set is meant to fail).

## What Inspect standardizes

- **Task/solver/scorer separation.** Dataset loading (`json_dataset` + `record_to_sample`),
  the system-under-test (solver), and judging (scorer) are separate registered components.
  run.py's `evaluate()` interleaves all three in one loop.
- **Logs as artifacts.** Every run writes a self-contained `.eval` file (transcripts, judge
  prompts, scores, token usage) into `evals/inspect-ai/logs/` (via `INSPECT_LOG_DIR` in `.env`); `inspect view` gives a browsable UI over all
  runs for free — roughly what `eval_results` + the :5003 trace viewer do, without owning
  a DB or a viewer.
- **Parallelism, retries, rate limits.** `--max-connections` gave a ~5x wall-clock win with
  zero code; HTTP retries and provider rate-limit handling are built in. run.py has none.
- **Resume.** An interrupted run prints an `inspect eval-retry <log>` command that reruns
  only unfinished samples. Our only equivalent is judge.py's skip-by-id JSONL resume.
- **Provider abstraction.** The judge model is a CLI flag (`--model openrouter/...`) — swapping
  judge provider needs no code. (Generation is deliberately NOT behind this — see trade-off.)
- **Selection ergonomics.** `--limit`, `--sample-id 59,73` for free; run.py has `--split/--limit` only.
- **Multi-metric scores.** Dict-valued `Score` + glob metrics (`{"*": [mean(), stderr()]}`)
  reproduce our per-dimension pass-rate table, with stderr added.

## What the hand-rolled harness has that Inspect doesn't (out of the box)

- **Config fingerprinting.** run.py hashes retrieval + gen prompt + judge rubric into a
  12-char content hash on the run record; Inspect logs task args but doesn't fingerprint
  arbitrary system-under-test config living outside its model abstraction.
- **Cross-run regression gating.** `check_regression.py` compares per-dimension pass rates
  against `baseline_metrics.json` and fails CI on a >0.15 drop. Inspect scores single runs;
  cross-run deltas (run.py's `print_summary` prev-run table) are yours to build.
- **Structured judge output.** instructor + Pydantic `Verdict{rationale, passed}` with
  re-ask on parse failure. The Inspect-idiomatic scorer is rationale-then-`VERDICT:` regex —
  works, but is string parsing; DeepSeek also spent ~2.8k reasoning tokens on judge calls
  that the hand-rolled `REASONING_OFF` extra_body suppresses (no obvious Inspect knob at
  the generate() call for OpenRouter reasoning).
- **Token-priced cost per run.** judge_db.py prices usage into a per-item `cost` column.
  Inspect counts tokens but only for calls it makes — our generation happens inside the
  solver thread, invisible to it.
- **Judge alignment + data flywheel.** `judge_vs_human.py` (TPR/TNR vs human labels) and
  `sample_traces.py` (Langfuse → candidate_eval_questions) have no framework equivalent;
  they're the actual moat.

## Trade-off observed

Evaluating a production RAG system means the solver calls the real pipeline
(`covered/retrieve/answer` in a thread), so Inspect can't log, cache, retry, or price the
generation half — its value concentrates in judging, parallelism, logging, and run
ergonomics. Routing generation through Inspect's model API would light all that up but
would evaluate a different system than production.

## Gotchas hit

- Dict-valued scores must have **identical keys across samples** (`results.py` raises
  otherwise); `float("nan")` marks a dimension not-applicable and is excluded from metrics.
- Inspect's OpenRouter provider requires `openai>=2.45.0` (bumped in pyproject).
- `.eval` logs dedupe long strings into `attachment://` pointers — resolve via the sample's
  `attachments` map when grepping dumps.
- Task file imports `rag`/`evals` — needs the repo root on `sys.path` (2-line shim at top).
