# RAG Failure Analysis

_Date: 2026-06-12 · System: naive RAG (voyage-4 embeddings, top-5 cosine, Sonnet 4.6), corpus after timecode cleanup (634 chunks)._

## Method

Two evidence sources, no guessing:
- **Retrieval** — `metrics.py` over the 19 questions with gold `expected_chunk_ids`: where did the gold chunk rank?
- **Generation** — human grading of all 25 RAG answers (correct / partial / wrong / hallucinated). Result reported: **all answers good.**

## Failure counts

| Category | Definition | Count |
|---|---|---|
| Retrieval miss | Right chunk never in top-5 | **1** (Q24) |
| Retrieval noise | Wrong chunks crowded the right one *out* of top-5 | **0** |
| Generation failure | Right chunk retrieved, bad answer anyway | **0** |
| Hallucination | Answer asserts content not in the chunks | **0** |

**The system is not meaningfully broken.** This is the expected result *after* the timecode cleanup — that change already fixed the large retrieval failures (e.g. the music question went from "no answer" to a clean hit, nearest distance 0.94 → 0.39).

## The one retrieval miss — Q24 (typo robustness)

Question: `how dose inerdance help with addction and dopmaine reroute??`

- Gold chunks `[1386, 1410, 1459, 1584, 1671]` (dopamine + addiction) did **not** appear in top-5.
- Retrieved instead: chunks at distance **0.556–0.633** — much farther than a normal query (~0.40). The misspellings (`dopmaine`, `addction`, `inerdance`) moved the query vector away from the right region.
- **Recovered anyway:** retrieval still pulled *adjacent* addiction/Day-1 chunks, so the generated answer was substantive and graded good.

So this is a **ranking degradation from lexical noise**, not an answer failure. It's the only place pure vector search showed brittleness.

## Non-failures worth noting (not broken, but watch)

- **Sub-optimal ranking:** Q5, Q6, Q14, Q22 ranked the gold at #2 (still hit). Harmless here; would matter more at top-1 or top-3.
- **Loose gold inflates the metric:** Q9 had 140 "gold" chunks, Q8 86, Q11 35 — keyword-auto-labeled, far too broad. recall@5 = 0.95 is **optimistic**; the trustworthy tight-gold subset was ~0.89.

## What's actually worth fixing (prioritized)

1. **Harden the eval, not the RAG.** The reason there are "no failures" is partly that the eval is small (25 Q) and the gold labels are loose. Before chasing RAG improvements, **tighten `expected_chunk_ids`** (especially Q8/9/11) and **grow the question set** with harder cases. Otherwise you can't detect the regressions you're trying to prevent. _Highest value._
2. **Lexical/typo robustness (Q24).** The only real RAG weakness. Fix = hybrid retrieval (keyword/BM25 + vector) or a query-normalization step. _Low priority — it didn't even break the answer; only do this if real users type messily._
3. **Do not touch generation.** Zero generation/hallucination failures. The system already refuses no-answers and resists distractors. Spending effort here would be fixing what isn't broken.

## Honest caveat

"No failures" is a statement about *this* eval, not proof of a great system. The clean result could mean (a) the cleanup genuinely fixed the big problems — true, it did — and (b) the eval isn't yet hard enough to expose the next tier of failures. Item 1 above addresses (b).
