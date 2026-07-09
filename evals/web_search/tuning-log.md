# Tuning log — web search agent

Protocol: attack the biggest failure category, one change at a time, re-run only
the affected questions (`run_baseline.py --ids ... --tag ...`), judge with the same
tagged flow, diff per-fact verdicts against the baseline judgments. Baseline
artifacts (`results.jsonl`, `judgments.jsonl`, `baseline_metrics.json`) are immutable.

Standing caveats: each re-run is a single sample of a stochastic agent on a live
web — a flipped verdict is evidence, not proof (during plumbing smoke-testing,
q4 scored `correct` *before any change*). Judge is deepseek-v4-flash with the
known self-agreement/weak-judge caveats from the baseline.

---

## Round 1 — target: synthesis_error (×7, biggest category)

**Diagnosis** (from `judgments.jsonl` failure rationales, 2026-07-08): in 6 of 7
rows (q4, q13, q14, q15, q17, q19 — q2 is premature_stop-dominated) the missing
fact was demonstrably *in the trajectory* but absent from the final answer:
detail dropped at write-up time, not miscalculation. The exercise's stock fixes
(3-source minimum, verification search) don't match this mechanism and were not
used.

### Change 1 — answer-completeness instruction (KEPT)

Added one line to `SYSTEM_PROMPT` answer requirements
(`rag/query/web_search_agent.py`):

> - Include every concrete detail you encountered that bears on the question —
>   dates, numbers, names, titles, records, "firsts" — even secondary ones.
>   A research answer errs on the side of completeness, not brevity.

Hypothesis: recovers omissions where the detail already survived into the
transcript. Re-ran `--ids 4,13,14,15,17,19 --tag round1a` (~$0.02 agent + judge).

| Q | baseline | round1a | facts flipped |
|---|----------|---------|---------------|
| 4 | partial | partial | none — the "4 Oscars / Best Actor" fact never entered the trajectory (agent has no reason to research it; see eval-design note) |
| 13 | partial | partial | none — "first opener win" again omitted despite being on fetched pages |
| 14 | partial | **correct** | WWDC dates missing→present; Siri chatbot detail missing→present |
| 15 | partial | **correct** | Levinson lead-independent-director missing→present |
| 17 | partial | wrong | not a prompt regression — ground-truth drift (see below) |
| 19 | partial | partial | churn: skeptics-estimate present→missing, others unchanged |

**Measured effect: 2 of 6 recovered (q14, q15), 3 unchanged, 0 true
regressions.** Estimated overall vs baseline: 12/20 → ~14/20 (single-sample).
Decision: **keep**.

### Change 2 — distiller prompt (NOT APPLIED)

Planned only if omissions persisted where distillation dropped the detail before
synthesis. Checked the round1a transcripts via Langfuse: for q13 and q19 the
key phrases ("first ... win", "accurate"/"lack") are present in the fetch_page
outputs that entered the transcript (4–7k chars, several below the 1500-token
distill threshold entirely). The distiller is not the blocker → change not
supported by data → not applied.

### Findings beyond the prompt (for future rounds / eval-set v2)

- **Ground-truth drift (q17, Saturn):** the agent found IAU announcements from
  2026 with a newer moon count; the eval facts froze the March-2025 state
  (274, "some 2026 sources say 285"). The judge now reads the agent's current
  count as a contradiction. The eval question needs its facts refreshed or
  rephrased date-anchored ("as of March 2025..."). Known risk from eval-set
  design, now materialized.
- **Bonus-fact grading (q4, q13):** some eval facts aren't implied by the
  question text (Sinners' win count when the question asks about nominations +
  cinematographer; Mexico's first-opener-win when the question asks result +
  disciplinary record). The agent has no reason to research them. Either the
  questions should ask for them, or these facts should be marked optional.
- **Residual write-up selectivity (q19):** completeness instruction helped
  elsewhere but flash still drops quotable specifics it read. Candidate for a
  later round: require a short "key specifics" recap before the final prose, or
  judge-visible drafts.

**Round 1 net: synthesis_error 7 → est. 3–4; overall est. 12/20 → ~14/20.
One prompt line added; nothing else changed.**

---

## Round 2 — target: premature_stop (×4 baseline; q15 already recovered)

**Diagnosis:** fetches are slow and the wall-clock check runs only between LLM
rounds, so nominal-60s runs actually consumed 65–94s — with the final answer
squeezed through the degraded best-effort path, which failed outright on q2/q7
(markup leak on both retries → "Could not produce an answer").

### Change 1 — MAX_SECONDS 60 → 90 (KEPT)

Deliberate reversal of the earlier revert to 60, approved via plan: the 60s
budget was already leaky (runs spent up to 94s), but the overshoot fed tool
rounds, not the answer. Re-ran `--ids 2,7,19 --tag round2a`:

| Q | baseline | round2a | note |
|---|----------|---------|------|
| 2 | wrong | **correct** | finally fetched the actual LKML mail; all 3 facts present |
| 7 | wrong | wrong | still "Could not produce an answer" (see change 2 + residual) |
| 19 | partial | **correct** | census deep-dive completed within budget |

2 of 3 recovered. Observed side effect: with 90s the leaky check now lets runs
stretch to 110–200s wall (multi-fetch rounds); acceptable on flash costs but
worth a strict in-loop check if latency ever matters.

### Change 2 — best-effort hardening (KEPT, but didn't fix q7)

`_best_effort_answer`: 3 attempts (was 2) + on total failure salvage the last
substantive assistant narration from the transcript. Re-ran `--ids 7 --tag
round2b`: still "Could not produce an answer" — there was no narration to
salvage (flash's assistant turns were pure tool calls) and all three attempts
leaked markup. Kept anyway: strictly-better fallback, no regression risk.

### Residual: q7 diagnosed at the time as needing a different change

The round2b run targeted the *correct* filings (MSFT FY2025, NVDA FY2026 on
sec.gov) — queries and source selection improved with the date-aware prompt —
but SEC 10-K pages are huge and `MAX_PAGE_TOKENS=4000` truncates them before
the Human-Capital section, so the employee counts never enter context. No
prompt fixes that. Candidate for round 3: feed the distiller more of the raw
page (e.g. distiller input cap ≫ transcript cap) or fetch-with-offset.
Emerging failure category candidate: `source-too-large-for-context`.

---

## Gate measurement — full 20-question re-run (2026-07-08, tag `gate1`)

Eval-set maintenance before the run, logged transparently: q17's facts were
date-anchored (the world moved — 2026 MPC announcements raised Saturn's count
past the frozen March-2025 numbers; the amended facts accept any current count
that explains the recognition-date dependence). q4/q13's bonus-fact issue was
deliberately NOT softened.

Gate metric (defined before the run): **correct-and-cited** = judged `correct`
(all facts present, mechanical rule) AND ≥1 markdown citation link (links are
trace-verified by the agent at answer time).

**Result: 16/20 correct-and-cited = 80% → GATE PASSED (≥70%).**
(16 correct, 4 partial, 0 wrong | fact recall 56/60 | metrics_gate1.json)

Verdict flips vs baseline, both directions (honesty note — single-sample runs
churn): recovered q2, q7, q13, q14, q15, q17, q19 → correct; regressed q1, q3
(baseline correct → partial this run; q3 lost the "not particularly seriously"
quote to an early stop, q1 dropped one sub-fact). Residual partials: q1, q3,
q4, q19 — failure labels: synthesis_error ×3, premature_stop ×3,
wrong_source_selection ×1.

Audits of surprising verdicts (flash-judge caveat): q7's `correct` is genuine —
the answer states MSFT 228,000 (confirmed unchanged through FY2025), NVDA
42,000 (FY ended 2026-01-25), difference 186,000; evidence quotes verified
against the answer text. q17's `correct` is genuine under the amended facts —
and the answer is better-researched than the baseline's (explains 274→285 via
the March 2026 MPC update).

Caveats on the 80%: single stochastic sample (the q1/q3 regressions show
±1–2 question churn is normal); judge is flash (same model family as the
agent); q17 counts partly because its facts were amended — without the
amendment the gate number would likely be 15/20 (75%), still passing.

**Config locked at gate pass:** the agent as of this commit — system prompt
(method + completeness line), date injection, tool descriptions, distillation
(flash, >1500 tokens), citation verification (2 retries), budgets 10 iterations
/ $0.50 / 90s, best-effort fallback (3 attempts + transcript salvage).

---

## Self-critique A/B (2026-07-09, tags `crit_on` / `crit_off`)

**The change:** one optional review pass (`SELF_CRITIQUE` flag) — when the model
first produces a draft answer, a critique prompt asks it to re-read the draft
against the gathered evidence, flag unsupported claims, and research any gaps
before finalizing. Fires once per run; runs under the normal budgets.

**The experiment:** 10 questions (the gate's 4 partials + 2 previously-fragile +
4 stable corrects), both arms run fresh the same day, judged identically.

| | critique ON | critique OFF |
|---|---|---|
| correct-and-cited | **7/10** | **6/10** |
| fact recall | 27/30 | 24/30 |
| mean wall time | **162s/question** | 125s/question |
| total cost | $0.027 | $0.031 (≈ equal) |
| runs ending on the time limit | **10/10** | 5/10 |
| critique actually fired | 6/10 (rest hit the limit before any draft) | — |

Per-question flips: critique helped q3, q7, q13 (each gained facts after the
review pass — q13-on verifiably searched to confirm the red-card record
post-critique); critique hurt q16 (284s of source-thrash → wrong) and q19
(correct in the off arm). Net +1.

**Measured verdict: self-critique did NOT reliably improve correctness here.**
The +1 correct (and +3 fact recall) is smaller than the ±2-question run-to-run
churn we've measured between *identical* configs (baseline→gate flipped q1/q3
with no changes), so it cannot be distinguished from noise at N=10. The effects
that ARE unambiguous: +30% mean latency, and the review pass turns every run
into a time-limit exhaustion (10/10 vs 5/10) — at a 90s budget the critique
doesn't get room to do its job (it only fired in 6 of 10 runs at all).

Both proposed mechanisms were observed in the traces — critique genuinely
triggers gap-filling research AND genuinely burns budget into thrash. They
roughly cancel.

**Decision: `SELF_CRITIQUE = False` by default** (code kept behind the flag).
If revisited: pair it with a larger time budget and N≥30 per arm to detect a
real effect, or trigger critique only when the draft cites <2 sources.
