# Archived evaluation runs

These runs are kept for documentation. They are not comparable with the final before/after runs because they
were scored with earlier versions of the judge. All runs used the same chatbot configuration
(top_k=4, min_similarity=0.5, prompt v1, gpt-4o-mini) and the 44-question dev set.

## 01 — judge v1: judge lists "unsupported claims" directly
File: `01_judge-v1-direct-flags__baseline__dev.json`

Faithfulness was 80.0%. A manual check showed that 6 of the 8 claims flagged as unsupported were in fact in the
retrieved passages (for example clove oil, the rubber dam, and private emergency clinics). The judge treated
anything beyond the short list of reference facts as unsupported. Reason for change: too many false positives.

## Lost runs — judge v1 code, fingerprint 3ba1c54c1490 (files deleted)
These two runs were scored after I had tried to switch to the quote-based judge, but the new `judge.py` had not
been saved to disk, so they were really judge v1 again. The result files were deleted before the instruction to
keep all results. Their summaries, copied from the console output:

| Metric | baseline | baseline_rerun |
|---|---|---|
| Retrieval Hit@4 | 100.0% | 100.0% |
| Retrieval Recall@4 | 96.2% | 96.2% |
| Retrieval MRR | 0.97 | 0.97 |
| Retrieval Precision@4 | 28.1% | 28.1% |
| Answer accuracy | 92.5% | 95.0% |
| Correctness (1-5) | 4.55 | 4.62 |
| Faithfulness | 85.0% | 80.0% |
| Completeness | 87.7% | 88.5% |
| False refusals | 2.5% | 2.5% |
| Correct refusals (out of scope) | 100.0% | 100.0% |
| Leaks | 2.3% | 2.3% |

## 02 — judge v2: judge must quote the context; quote checked as an exact substring
Files: `02_judge-v2-exact-quote__baseline__dev.json`, `02_judge-v2-exact-quote__baseline_rerun__dev.json`
(fingerprint 2274f2b72b00)

Faithfulness was 85.0% and 87.5%. Many flags were still false positives, for these reasons:
- The judge joined text across bullet lines, for example "Tiotandvård does not cover X-rays" from a heading and
  a bullet, so the quote was not an exact substring.
- The judge garbled å/ä/ö in quotes (for example "Sjögren" was written with a broken escape character).
- Sentences such as "I don't have specific information about Umeå" were counted as unsupported facts, although
  they are the correct partial-answer behavior.

## Current judge (v3), used for all comparable runs
- A quote counts as supported if at least 90% of its words occur in one retrieved passage.
- Quotes shorter than 3 words don't count as support.
- Statements that the chatbot lacks information are not scored as factual claims; mentions of "the context" are
  still caught by the separate leak check.
