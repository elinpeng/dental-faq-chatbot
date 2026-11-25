"""Compare evaluation runs side by side, only if they were scored with identical criteria.

Example (run from the project root):
    .venv/bin/python -m evals.compare evals/results/baseline__dev.json evals/results/refined__dev.json
"""
import json
import sys
from pathlib import Path

ROWS = [
    ("retrieval_hit_at_k", "Retrieval Hit@k", "pct"),
    ("retrieval_recall_at_k", "Retrieval Recall@k", "pct"),
    ("retrieval_mrr", "Retrieval MRR", "num"),
    ("retrieval_precision_at_k", "Retrieval Precision@k", "pct"),
    ("avg_chunks_retrieved", "Avg chunks retrieved", "num"),
    ("answer_accuracy", "Answer accuracy", "pct"),
    ("answer_correctness_avg", "Correctness (1-5)", "num"),
    ("faithfulness", "Faithfulness", "pct"),
    ("completeness", "Completeness", "pct"),
    ("false_refusal_rate", "False refusals (lower is better)", "pct"),
    ("correct_refusal_rate", "Correct refusals, out of scope", "pct"),
    ("leak_rate", "Leaks (lower is better)", "pct"),
]


def fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if kind == "pct" else f"{value:.2f}"


def main(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    fingerprints = {(r["config"]["set"], r["config"]["eval_fingerprint"]) for r in runs}
    if len(fingerprints) > 1:
        sys.exit(f"Not comparable: runs use different question sets or evaluation criteria {fingerprints}")

    cfg = runs[0]["config"]
    print(f"Set: {cfg['set']} ({cfg['n_questions']} questions)  judge: {cfg['judge']}  "
          f"eval fingerprint: {cfg['eval_fingerprint']}\n")
    headers = [f"{r['config']['tag']} (k={r['config']['top_k']}, sim={r['config']['min_similarity']}, "
               f"prompt {r['config']['prompt_version']})" for r in runs]
    print("| Metric | " + " | ".join(headers) + " |")
    print("|---|" + "---|" * len(runs))
    for key, label, kind in ROWS:
        if all(key in r["summary"] for r in runs):
            print(f"| {label} | " + " | ".join(fmt(r["summary"][key], kind) for r in runs) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
