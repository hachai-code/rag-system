# evals/ — map

Offline evaluation for the innerdance RAG. Read [`DESIGN.md`](DESIGN.md) for the
*why* (Hamel's three-level framework, priority order); this file is the *where*.

Everything is split by **what it evaluates**. Cross-cutting pieces (the test
bank, the orchestrator, the CI gate, the shared schema) sit at the top; the two
subsystems and the review UIs each get a folder.

```
evals/
  DESIGN.md              the plan and rationale
  README.md              this map
  eval_set.jsonl         the shared test bank — questions, ideal answers, gold chunk ids
  traces.jsonl           eval traces for viewers/trace_viewer.py
  langfuse_traces.jsonl  production traces for viewers/langfuse_viewer.py
  schema.py              shared Pydantic row schema
  run.py                 one-command run: retrieve → answer → judge, logged to Postgres
  check_regression.py    CI gate: fail a PR if a category's pass rate drops too far
  test_assertions.py     Level-1 deterministic checks (uv run pytest)
  baseline_metrics.json  blessed pass rates the regression gate compares against
  configs/               versioned run knobs (baseline, vector, hype, multi_query, prose)

  search/                evaluate RETRIEVAL
    metrics.py             recall@5, MRR  →  data/metrics_log.jsonl
    compare_retrieval.py   vector-vs-keyword diagnostic on hard queries
    gen_questions.py       synthesize tight-gold questions from corpus chunks
    show_questions.py      pretty-print the synthetic set
    data/                  synthetic_questions.jsonl, metrics_log.jsonl

  answer/                evaluate the ANSWERING path (retrieval + generation)
    judge.py               LLM-as-judge, one narrow Opus call per rubric dimension
    judge_db.py            same judge, persisted to Postgres (eval_runs + eval_results)
    judge_vs_human.py      judge↔human agreement  →  analysis/judge_metrics.json
    gen_eval.py            fill eval_set.jsonl with draft RAG answers for grading
    eval_answers.py        end-to-end answer dataset  →  data/answer_feedback.jsonl
    deepeval_run.py        off-the-shelf RAG-triad scores (DeepEval)
    rubrics.md             the A–F judge rubric
    data/                  human eval, judgments, answer feedback
    analysis/              failure taxonomies, judge validation write-ups, judge metrics

  viewers/               human review UIs (open in a browser / localhost)
    grade.py               grade eval_set answers (correct/partial/wrong/hallucinated)
    grade_answers.py       grade answer/data/answer_feedback.jsonl (up/down)
    trace_viewer.py        open-coding over eval traces (free-text notes)
    langfuse_viewer.py     open-coding over REAL production traces from Langfuse
    labelling/             standalone FastHTML labelling app

  web_search/            evaluate the WEB SEARCH agent (separate from the RAG;
    eval_set.jsonl         its own test bank — questions, must_contain, facts
    run_baseline.py        run the agent over the set, resumable, → data/results.jsonl
    judge.py               two-stage judge: grade vs facts, then failure taxonomy
    tuning-log.md          what was tried and what moved the numbers
    data/ analysis/        results + judgments per run; metrics per run

  notes/                 failure-analysis.md, chunking-experiments.md
```

## Running

Run everything as modules from the repo root so `import rag` resolves:

```bash
uv run pytest                                              # Level-1 assertions
uv run python -m evals.search.metrics [label]             # retrieval metrics
uv run python -m evals.run --config evals/configs/baseline.json
uv run python -m evals.check_regression --config evals/configs/baseline.json --split dev
uv run python -m evals.viewers.trace_viewer               # then open the printed URL
```

Data files are anchored relative to each script (`Path(__file__)`), so cwd
doesn't matter. `eval_set.jsonl` lives at the top because both `search/` and
`answer/` read it.
