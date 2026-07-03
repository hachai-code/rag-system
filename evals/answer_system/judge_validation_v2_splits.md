# Judge Validation v2 — dev vs. test split

_Companion to `judge_validation_v2.md`. Genuine verdicts only — the 6 soft-refusals (judge hallucinated "garbled") are excluded, so this is the 31-pair set partitioned by the `split` field._

Why this matters: rubric changes should be tuned on **dev** and confirmed on held-out **test**. The pooled v2 numbers mix the two, so a fix measured on the pooled set is partly fit to data you'd want kept blind. Per-split n is ~19, so treat each κ as a noisy floor.

## Headline

| | pooled v2 | dev | test |
|---|---|---|---|
| pairs scored (n) | 31 | 10 | 11 |
| refused (missing) | 7 | 0 | 2 |
| agreement | 0.645 | 0.8 | 0.455 |
| Cohen's κ | 0.241 | 0.615 | 0.029 |
| disagreements | 11 | 2 | 6 |

## dev split

n=10  ·  agreement=0.8  ·  κ=0.615  ·  refused=0
confusion: HP-JP=4 HP-JF=0 HF-JP=2 HF-JF=4

| Dim | Judged | Agree | Agree % | Refused |
|-----|--------|-------|---------|---------|
| A — Security & IP | 1 | 1 | 100% | 0 |
| B — Retrieval | 1 | 1 | 100% | 0 |
| C — Generation | 4 | 3 | 75% | 0 |
| D — Grounding | 1 | 0 | 0% ⚠️ | 0 |
| E — Formatting | 3 | 3 | 100% | 0 |

Disagreements: 47C(F→P), 68D(F→P)

## test split

n=11  ·  agreement=0.455  ·  κ=0.029  ·  refused=2
confusion: HP-JP=2 HP-JF=1 HF-JP=5 HF-JF=3

| Dim | Judged | Agree | Agree % | Refused |
|-----|--------|-------|---------|---------|
| A — Security & IP | 4 | 3 | 75% | 0 |
| B — Retrieval | 3 | 0 | 0% ⚠️ | 0 |
| C — Generation | 3 | 2 | 67% | 0 |
| D — Grounding | 1 | 0 | 0% ⚠️ | 0 |
| E — Formatting | 0 | 0 | — | 1 |
| F — Corpus/Data | 0 | 0 | — | 1 |

Disagreements: 54A(F→P), 2B(F→P), 34B(F→P), 60B(P→F), 32C(F→P), 11D(F→P)

## Read

- **B fails out-of-sample too.** dev 1/1, test 0/3 — the retrieval-rubric problem isn't a dev artifact; it reproduces on held-out test, so it's worth fixing.
- **κ per split is noisy** (n≈19, FAIL-heavy marginals) — same base-rate paradox as the pooled number; read the per-dimension agreement, not the single κ.
- **After rewriting the B rubric, tune on dev and report the change on test** to avoid fitting the fix to the same data that diagnosed it.

Artifacts: `judge_metrics_v2_dev.json`, `judge_metrics_v2_test.json`. Pooled report unchanged in `judge_validation_v2.md`.
