"""Summarize repeated runs of one configuration and list failures that recur across runs.

Example (run from the project root):
    .venv/bin/python -m evals.analyze evals/results/baseline__dev.json evals/results/baseline_rerun__dev.json
"""
import json
import sys
from pathlib import Path
from statistics import mean

KEYS = ["retrieval_recall_at_k", "retrieval_precision_at_k", "answer_accuracy", "answer_correctness_avg",
        "faithfulness", "completeness", "false_refusal_rate", "correct_refusal_rate", "leak_rate"]


def main(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    fingerprints = {r["config"]["eval_fingerprint"] for r in runs}
    if len(fingerprints) > 1:
        sys.exit(f"Runs were scored with different criteria: {fingerprints}")
    n = len(runs)
    print(f"{n} runs, eval fingerprint {fingerprints.pop()}\n")
    print(f"{'metric':<26}" + "".join(f"{'run ' + str(i + 1):>9}" for i in range(n)) + f"{'mean':>9}{'range':>9}")
    for k in KEYS:
        values = [r["summary"][k] for r in runs]
        print(f"{k:<26}" + "".join(f"{v:>9.3f}" for v in values) + f"{mean(values):>9.3f}{max(values) - min(values):>9.3f}")

    print("\nFailures (count = number of runs where it happened):")
    by_id = [{row["id"]: row for row in r["rows"]} for r in runs]
    for qid, first in by_id[0].items():
        rows = [b[qid] for b in by_id]
        if first["type"] == "out_of_scope":
            wrong = [r for r in rows if not r["correct"]]
            if wrong:
                print(f"\n{qid} [out of scope] {first['question']}\n   answered instead of refusing {len(wrong)}/{n}: {wrong[0]['answer'][:200]}")
            continue
        issues: dict[str, list] = {}
        for r in rows:
            checks = [("false refusal", r["refused"], r["answer"]),
                      ("leak", bool(r["leaks"]), r["answer"][-150:]),
                      ("unsupported", bool(r["unsupported_claims"]), r["unsupported_claims"]),
                      ("incorrect", bool(r["incorrect_claims"]), r["incorrect_claims"]),
                      ("missed facts", bool(r["missed_facts"]), r["missed_facts"])]
            for name, failed, detail in checks:
                if failed:
                    issues.setdefault(name, []).append(detail)
        recall = first["retrieval"]["recall"]
        if issues or recall < 1:
            print(f"\n{qid} {first['question']}  (retrieval recall {recall:.0%})")
            if recall < 1:
                print("   retrieved: " + " | ".join(x[:60] for x in first["retrieved"]))
            for name, details in issues.items():
                print(f"   {name} {len(details)}/{n}: {str(details[0])[:260]}")


if __name__ == "__main__":
    main(sys.argv[1:])
