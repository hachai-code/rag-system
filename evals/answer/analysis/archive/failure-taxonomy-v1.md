# Answer-System Failure Taxonomy

_Date: 2026-06-19 · Source: human review of RAG answers, coded in `rag_system_human_eval.txt`. Formalizes the labels into a stable set of categories the judges and eval set (`rag_system_human_eval.jsonl`) are built on._

This is the **generation/answer-quality** taxonomy. It complements `../failure-analysis.md`, which covers retrieval-layer failures (recall, ranking, typo robustness) measured against gold chunks.

## How it was built

Standard qualitative coding, three passes, no guessing — every category traces back to a real reviewer label:

1. **Raw labels** — free-text notes left while reading answers ("hallucinated bibliography", "just repeats the corpus", "needs citation markers").
2. **Open codes (12)** — the labels clustered into single-purpose failure modes.
3. **Axial categories (5, A–E)** — open codes folded into the categories that become one binary judge each. See `rubrics.md`.

The value of keeping all three layers: a label tells you *what a human noticed*, an open code tells you *the failure mode*, an axial category tells you *which judge owns it*. Collapsing too early loses the layer signal (a trace can pass retrieval but fail security — see code 3 below).

## Open codes

| # | Open code | The failure | Axial parent |
|---|-----------|-------------|--------------|
| 1 | Source/corpus concealment | Frames answers as "from a document" / exposes that documents exist | A |
| 2 | IP / extraction resistance | Lists or **hallucinates** source documents when asked to extract them | A |
| 3 | Speaker attribution | Attributes a student's words to the teacher (retrieval) **or** names a speaker in the output (generation) | B (retrieval) + A (output) |
| 4 | Scientific validation | States neuro/physiology claims without grounding them | D |
| 5 | External augmentation | Fails to enrich with outside context when the corpus alone is thin | D |
| 6 | Recall / completeness | Misses relevant chunks, so the answer is incomplete | B |
| 7 | Topic precision | Pulls adjacent-but-wrong-topic content (kundalini for an innerdance question) | B |
| 8 | Synthesis vs. verbatim | Parrots corpus text instead of reasoning over it | C |
| 9 | Task-intent routing | Treats a "create/draft" request as a flat lookup | C |
| 10 | Citation formatting | Missing citation markers | E |
| 11 | Style / terminology | Violates surface rules (e.g. "innerdance" lowercase, joined) | E (+ A framing) |
| 12 | Holistic quality | Catch-all human rating of overall answer goodness | C |

## Axial categories

Each is a single binary judge in `rubrics.md`. Severity drives prioritization.

| Code | Category | Folds in | Layer | Severity |
|------|----------|----------|-------|----------|
| **A** | Security & IP Protection | 1, 2, 3-output, 11-framing | Generation (output filter) | **Safety-critical** — legal/business risk |
| **B** | Retrieval Quality | 6, 7, 3-retrieval | Retrieval | Quality |
| **C** | Generation Quality | 8, 9, 12 | Generation | Quality |
| **D** | Grounding & Factual Validation | 4, 5 | Retrieval + generation | **Safety-critical** — health-adjacent harm |
| **E** | Formatting & Conventions | 10, 11 | Output format | Quality (cheapest fix) |

### Two deliberate design choices

- **Code 3 spans A and B on purpose.** Diarize *internally* so retrieval gets the right speaker's turn (B); strip speaker identity *externally* so the output reads as unified corpus knowledge (A). A trace can pass B (right chunk) and still fail A (names the speaker). Keep them as two judge questions, not one.
- **Codes 6, 8, 12 are all "answer quality" but stay split.** 6 is a *retrieval* hypothesis, 8 is a *generation* failure mode, 12 is a *holistic* human rating. Merging them would hide which layer broke.

## Severity & prioritization

**A and D are the safety-critical pair.** A is IP/legal exposure (the "extract the corpus" request is effectively a jailbreak/extraction attack). D reaches potentially vulnerable users — the eval set's health cluster (blood pressure, endometriosis, insomnia, addiction, autism) makes unverified medical-adjacent claims the highest-harm generation failure. B, C, E are quality-of-life. Fail A or D hard even when the rest of the answer is excellent.

## Class-imbalance caveat

Primary-code distribution of the 50-item eval set (`rag_system_human_eval.jsonl`):

| Category | A | B | C | D | E |
|----------|---|---|---|---|---|
| Primary items | 5 | 3 | 31 | 11 | 0 |
| Any-tag items | 5 | 4 | 34 | 15 | 1 |

The set is **C-heavy and starved on A, B, E** — E has no primary items at all. Stratification keeps ratios right across the dev/test split, but those counts are too small to measure per-category judge TPR/TNR: you cannot estimate a true-negative rate for A with 5 examples, none of them deliberate red-team attacks. **Before trusting any category-level judge accuracy, grow A (extraction red-team set) and D (medical-adjacent), and add E examples.** This is the same "harden the eval first" conclusion as `../failure-analysis.md`.
