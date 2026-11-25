# Evaluation report: Dental FAQ chatbot

The chatbot was evaluated on predefined dental questions for retrieval relevance, answer accuracy, and response
completeness. Retrieval settings and prompts were then refined based on the results, and the final version was
verified on a separate test set that was not used during refinement.

**Result:** On the held-out test set, the refined chatbot improved completeness from 81.6% to 85.4% and
correctness from 4.59 to 4.70 (out of 5), and retrieval recall from 90.9% to 93.2%. It kept 100% correct
refusals of out-of-scope questions and 0% false refusals. Faithfulness and accuracy did not change measurably.

## 1. Setup

| Component | Setting |
|---|---|
| Knowledge base | 15 documents on dental care in Sweden, 143 chunks |
| Embeddings | nomic-embed-text-v1.5, run locally on CPU |
| Vector store | ChromaDB, cosine similarity |
| Chatbot LLM | gpt-4o-mini, temperature 0 |
| Judge LLM | gpt-4.1-mini, temperature 0 (a different, stronger model, to reduce self-preference bias) |

### Question sets
Both sets were written before any evaluation was run. For every question, the set lists the sections that should
be retrieved ("gold sections") and the facts a complete answer must contain. A script checked that every gold
section exists and that every number in the reference facts appears in its gold section.

| Set | Questions | Purpose |
|---|---|---|
| Dev (`dev_questions.json`) | 44: 38 answerable, 2 partly answerable, 4 out of scope | Finding weaknesses and tuning |
| Test (`test_questions.json`) | 26: 21 answerable, 1 partly answerable, 4 out of scope | Final check only, run after the final configuration was fixed |

The test questions are phrased differently from the dev questions: casual wording, missing punctuation,
symptoms described in everyday words, and edge cases such as a 68-year-old asking whether tiotandvård
covers an implant on a back molar (it does not).

### Metrics
| Area | Metric | How it is measured |
|---|---|---|
| Retrieval relevance | Hit@k, Recall@k, MRR, Precision@k | Automatically, against the gold sections |
| Answer accuracy | Accuracy, Correctness (1-5) | Judge score; an answer is accurate if the score is at least 4 and it contains no incorrect claims |
| Faithfulness | Share of answers with no unsupported claims | The judge must quote the retrieved text supporting each claim; code checks that the quote is really there |
| Response completeness | Share of reference facts covered | The judge decides covered/not covered for each reference fact |
| Refusals | False refusals, correct refusals | Answerable questions must not be refused; out-of-scope questions must be refused |
| Leaks | Share of answers that show citations, file names, or mention sources | Pattern check on the user-facing answer |

### Fair comparison
- The judge, the scoring code, and both question sets were frozen before the comparable baseline was run.
  `FROZEN_EVAL_HASHES.txt` holds their SHA-1 hashes, and they were verified unchanged throughout.
- Every result file records an evaluation fingerprint (a hash of the judge, scoring code, and question set).
  `compare.py` and `analyze.py` refuse to compare runs with different fingerprints. All dev runs below have
  fingerprint `551f9f0726db`, and all test runs have `07894137735b`.
- Only chatbot settings changed between runs: top_k and the prompt version.
- Each main configuration was run 3 times to measure run-to-run noise. Numbers below are means of 3 runs unless
  marked "1 run".

## 2. Judge calibration (before the baseline)
The first judge version could not be trusted, so it was fixed before any comparable run:

1. **v1: judge lists unsupported claims directly.** A manual check showed that 6 of the 8 flagged claims were in
   fact in the retrieved passages. The judge treated anything beyond the reference facts as unsupported.
2. **v2: the judge must quote the supporting context, checked as an exact substring.** This still gave false
   flags, because the judge joined text across bullet lines and garbled å/ä/ö in quotes.
3. **v3 (used for all results): word-coverage check.** A quote counts if at least 90% of its words occur in one
   retrieved passage. Quotes under 3 words don't count, and sentences saying the bot lacks information are not
   scored as factual claims.

Runs scored with v1 and v2 are kept in `results/archive/` with notes in `results/archive/ARCHIVE_NOTES.md`.

## 3. Baseline (dev set)
Configuration: top_k=4, min_similarity=0.5, prompt v1.

| Metric | Run 1 | Run 2 | Run 3 | Mean | Range |
|---|---|---|---|---|---|
| Retrieval Hit@4 | 100% | 100% | 100% | 100% | 0 |
| Retrieval Recall@4 | 96.3% | 96.3% | 96.3% | 96.3% | 0 |
| Retrieval Precision@4 | 28.1% | 28.1% | 28.1% | 28.1% | 0 |
| Answer accuracy | 95.0% | 95.0% | 95.0% | 95.0% | 0 |
| Correctness (1-5) | 4.68 | 4.63 | 4.70 | 4.67 | 0.08 |
| Faithfulness | 97.5% | 95.0% | 95.0% | 95.8% | 2.5 |
| Completeness | 89.8% | 87.3% | 90.6% | 89.2% | 3.3 |
| False refusals | 2.5% | 2.5% | 2.5% | 2.5% | 0 |
| Correct refusals (out of scope) | 100% | 100% | 100% | 100% | 0 |
| Leaks | 2.3% | 2.3% | 2.3% | 2.3% | 0 |

The ranges show the noise between identical runs, about 3 points for completeness and 2.5 points for
faithfulness. A change has to be bigger than that to count.

## 4. Weaknesses found
Failures that happened in all 3 baseline runs:

| Weakness | Questions | Cause |
|---|---|---|
| Incomplete answers: the fact was in the retrieved passages but left out | d10, d13, d15, d23, d25, d27, d28 | The prompt asked for brief answers, so conditions and details were dropped. For d13 (knocked-out tooth) the bot used the short child section and ignored the more detailed section. |
| False refusal | d40 (antibiotic dose for an abscess) | When the exact thing asked for was missing, the bot refused completely instead of giving the related information ("antibiotics alone are not enough"). |
| Unneeded disclaimer that leaked | d37 | "…no specific information about the cost for a 50-year-old in the context provided": a detail that doesn't change the answer, and a mention of "context". |
| Outside knowledge added | d14 | "Cold sores spread through direct contact" is not in the knowledge base. |
| A second relevant section not retrieved | d21, d40, d12 | Only 4 chunks were retrieved. |

## 5. Refinement

### Retrieval settings (no LLM involved)
| top_k | Hit | Recall | MRR | Precision |
|---|---|---|---|---|
| 3 | 100% | 95.0% | 0.971 | 36.7% |
| 4 (baseline) | 100% | 96.3% | 0.971 | 28.1% |
| **5 (chosen)** | 100% | **98.8%** | 0.971 | 23.5% |
| 6 | 100% | 98.8% | 0.971 | 19.6% |
| 8 | 100% | 98.8% | 0.971 | 14.7% |

- **top_k 4 → 5.** The second relevant section for d21 and d40 is at rank 5. Beyond 5 there is no gain in recall,
  only more irrelevant text in the prompt.
- **min_similarity kept at 0.5.** The lowest top-1 similarity of an answerable question was 0.554 (mean 0.803), while out-of-scope
  questions reached 0.726 ("How do I fix my dog's broken tooth?"). Any threshold high enough to block those
  would also block real questions, so refusals are left to the prompt. The 0.5 threshold still stops
  completely unrelated questions such as a kanelbullar recipe (0.491) without calling the LLM.
- **d12 is not fixable by top_k.** One gold section is not in the top 8. The answer is still correct, because
  the section it does retrieve also says to go to the emergency room for facial swelling.

### Prompt changes
- **v2:** require every relevant detail (steps, conditions, exceptions, age limits, lists), combine overlapping
  passages, give related information when a specific detail is missing, mention missing details only when the
  user asked for them, and forbid explanations from the model's own knowledge.
- **v3:** v2 plus three fixes for what remained after v2:
  - mention what a symptom can be a sign of
  - say "I don't have information about…" instead of "I can't provide…"
  - refuse only if no passage is about the topic at all

  The wording change in the second fix is also the phrasing the frozen judge recognizes as an admission of missing
  information. Under v2, the judge counted "I can't provide the specific cost for Umeå" as an unsupported claim.

The new rules are written in general terms. Dev questions were not copied into the prompt as examples, so the
test set remains a fair check.

## 6. Before and after on the dev set
| Metric | Baseline k=4, v1 (3 runs) | k=5, v1 (1 run) | k=4, v2 (1 run) | k=5, v2 (3 runs) | **Final k=5, v3 (3 runs)** |
|---|---|---|---|---|---|
| Retrieval Recall@k | 96.3% | 98.8% | 96.3% | 98.8% | **98.8%** |
| Retrieval Precision@k | 28.1% | 23.5% | 28.1% | 23.5% | 23.5% |
| Answer accuracy | 95.0% | 97.5% | 97.5% | 97.5% | **97.5%** |
| Correctness (1-5) | 4.67 | 4.75 | 4.80 | 4.77 | **4.78** |
| Faithfulness | 95.8% | 97.5% | 95.0% | 94.2% | **96.7%** |
| Completeness | 89.2% | 90.8% | 94.0% | 92.4% | **92.9%** |
| False refusals | 2.5% | 2.5% | 2.5% | 2.5% | 2.5% |
| Correct refusals | 100% | 100% | 100% | 100% | **100%** |
| Leaks | 2.3% | 0% | 0% | 0% | **0%** |

- Most of the completeness gain comes from the prompt changes. The single k=4/v2 run scored 94.0%, slightly
  above the k=5/v2 mean, but one run is within the noise, so the evaluation does not show whether k=5 helps
  answer quality. It was kept for its better retrieval recall.
- The completeness ranges don't overlap. The lowest final run (92.5%) is above the highest baseline run (90.6%).
- d37 (leak and unneeded disclaimer) and the incomplete answers d10, d13, d27, d28 were fixed.

## 7. Final verification on the test set
The test set was run only after the final configuration was fixed. Nothing was changed after seeing these results.

| Metric | Baseline k=4, v1 (3 runs) | Final k=5, v3 (3 runs) |
|---|---|---|
| Retrieval Hit@k | 95.5% | 95.5% |
| Retrieval Recall@k | 90.9% | **93.2%** |
| Retrieval Precision@k | 30.7% | 25.5% |
| Answer accuracy | 95.5% | 95.5% |
| Correctness (1-5) | 4.59 (range 0.09) | **4.70** (range 0.05) |
| Faithfulness | 87.9% (range 9.1) | 89.4% (range 13.6) |
| Completeness | 81.6% (range 2.3) | **85.4%** (range 2.3) |
| False refusals | 0% | 0% |
| Correct refusals (out of scope) | 100% | 100% |
| Leaks | 3.8% | 3.8% |

- **Completeness (+3.8 points) and correctness (+0.11) carried over from dev to test.** The test gains are about
  the same size as on dev, and the completeness ranges don't overlap.
- **Faithfulness is inconclusive.** With 22 answerable questions, one answer moves the score by 4.5 points, and
  the runs varied by 9 to 14 points.
- **The test leak is a false alarm of the leak check.** In t21 the answer says "the source of infection must be
  treated", and the filter caught the word "source". No real leak occurred on the test set.
- **Test scores are lower than dev scores for both versions,** as expected for differently phrased questions.

## 8. Known limitations
- **Generic referral advice.** The final version sometimes adds advice like "consult a dental professional" that
  is not in the knowledge base (t01 in 3/3 runs, t09 in 2/3). It is harmless but breaks the grounding rule.
- **Vocabulary mismatch in retrieval.** "Eating ice cream hurts my teeth" (t16) never finds the sensitive-teeth
  sections, in either version. A fix would be adding everyday wording to the documents, or query rewriting.
- **Dose questions are refused.** "What dose of antibiotics should I take for a tooth abscess?" (d40) is refused
  in every run, despite two targeted prompt rules. gpt-4o-mini seems to refuse medication-dose questions. Refusing
  is safe, but the user misses the useful fact that antibiotics alone don't cure an abscess.
- **Missed warning sign.** For a mouth sore that hasn't healed in three weeks (d15), the answer says to get it
  checked but leaves out that it can be an early sign of oral cancer.
- **Small question sets.** 44 and 26 questions measure large effects only. Differences of a few points on
  faithfulness are within the noise.
- **The judge is an LLM.** Calibration removed its systematic false flags, but individual scores can still be
  wrong. All answers and judgements are saved in the result files for manual review.
- **The knowledge base itself was not fact-checked by this evaluation.** The evaluation measures whether answers
  match the documents, not whether the documents are correct. Several prices in the documents are rough estimates.

## 9. Reproduce
Run from the project root:
```
.venv/bin/python -m evals.sweep_retrieval --set dev
.venv/bin/python -m evals.run_eval --set dev --tag my_run --top-k 5 --prompt v3
.venv/bin/python -m evals.analyze evals/results/k5_v3_run1__dev.json evals/results/k5_v3_run2__dev.json evals/results/k5_v3_run3__dev.json
.venv/bin/python -m evals.compare evals/results/baseline__dev.json evals/results/k5_v3_run1__dev.json
shasum -c evals/FROZEN_EVAL_HASHES.txt
```

| Result files | Run |
|---|---|
| `results/baseline__dev.json`, `results/archive/baseline_rerun__dev.20261126-000150.json`, `results/baseline_rerun__dev.json` | Dev baseline, 3 runs |
| `results/k5_v1__dev.json`, `results/k4_v2__dev.json` | Dev, one change at a time |
| `results/k5_v2_run{1,2,3}__dev.json` | Dev, k=5 + v2 |
| `results/k5_v3_run{1,2,3}__dev.json` | Dev, final configuration |
| `results/test_baseline_run{1,2,3}__test.json`, `results/test_final_run{1,2,3}__test.json` | Test set |
| `results/archive/` | Earlier runs, kept as backups; never deleted or overwritten |
