# Chunking Experiments

_Date: 2026-06-12 · Goal: improve retrieval (recall@5 / MRR) by changing how the corpus is chunked and embedded._

## Headline result

**Two changes adopted, both measured wins, no regressions:**

| config | tight recall@5 | tight MRR | full recall@5 | full MRR | chunks |
|---|---|---|---|---|---|
| baseline (512 tok, no prefix) | 0.89 | 0.72 | 0.95 | 0.77 | 634 |
| + section/title prefix (512) | 1.00 | 0.78 | 1.00 | 0.79 | 634 |
| **+ prefix + 256 tok (adopted)** | **1.00** | **0.85** | **1.00** | **0.86** | 1538 |

- **Contextual prefix** (prepend `section — title` to each chunk's *embedded* text) raised recall@5 from 0.89 → 1.00 — it rescued the Q24 typo question, because the prefix `Maia — Day 1 Conditions` anchors the chunk's vector even when the query is misspelled.
- **Smaller chunks (256 vs 512)** raised MRR from 0.78 → 0.85 — more focused chunks match the query more precisely, so the right chunk ranks closer to #1. Recall was already at ceiling.

Both are now the defaults (`embed.py` always prefixes; `chunk.py` `CHUNK_SIZE=256`). The live DB already holds this config.

## A real methodology finding: ID-based gold is not re-embed-safe

The first prefix run reported `recall@5 = 0.00` everywhere — a false catastrophe. Cause: the gold labels were pinned to `chunks.id`, an auto-increment that gets **fresh values on every re-embed** (store does DELETE + INSERT). After re-embedding, the frozen gold IDs no longer existed.

**Fix:** the metric now recomputes the relevant-chunk set from **content keywords** against the *current* DB on every run (`metrics.gold_ids`). This is robust to re-embedding *and* re-chunking (where the chunk objects themselves change), which is exactly what a chunking experiment needs. Lesson: never key a retrieval eval on storage IDs that change when you re-index.

Comparable runs in `metrics_log.jsonl` are `baseline_kw`, `prefix`, `size256_prefix` (all use the fixed metric).

## What I did NOT do, and why

- **Semantic / heading-based splitting:** deprioritized. After the timecode cleanup the transcripts (12 of 15 docs, the bulk) have **no headings** to split on, and the EPUB's headings are already captured as chunk metadata. A structural splitter would only touch the one book, which already retrieves fine. Low expected value for the effort.
- **Larger chunks (768+):** the 512 → 256 trend (MRR up as size drops) points away from larger, so I didn't spend a run confirming the obvious regression. Going below 256 risks splitting answers across chunks and dropping recall — untested, a possible follow-up.

## Caveats

- **Recall is at a ceiling (1.00).** The eval can no longer discriminate on recall; only MRR has headroom. Further chunking gains can't be *measured* until the eval is harder / gold is tighter (the priority from `failure-analysis.md`).
- **256 changes generation context.** Retrieval improved, but smaller chunks mean less context per chunk fed to Claude (5×256 vs 5×512 tokens). Answer quality was graded good on the 512 config; it should be **re-graded** on 256 before fully trusting the end-to-end system. `eval_set.jsonl` draft answers and `grade.html` are now stale — re-run `gen_eval.py` then `grade.py`.
- Gold is still keyword-approximate; tightening it (failure-analysis item #1) would firm up these numbers.

## Bottom line

Measured improvement: **recall@5 0.89 → 1.00, MRR 0.72 → 0.85** on the discriminating tight-gold subset, from contextual prefixing + smaller chunks. The bigger takeaway is that the eval harness had to be fixed (ID → content gold) before any of this was trustworthy.
