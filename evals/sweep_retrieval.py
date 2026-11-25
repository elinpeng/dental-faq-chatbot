"""Retrieval-only sweep over top_k, plus the similarity gap between in-scope and out-of-scope questions.

No OpenAI calls. Uses the same retrieval metrics as run_eval.py.

Example (run from the project root):
    .venv/bin/python -m evals.sweep_retrieval --set dev
"""
import argparse
from statistics import mean

from evals.run_eval import load_questions, retrieval_metrics
from src.vectorstore import get_vectorstore

K_VALUES = [3, 4, 5, 6, 8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test"], default="dev")
    args = ap.parse_args()

    questions = load_questions(args.set)
    store = get_vectorstore()
    # Retrieve once at the largest k; smaller k is a prefix of the same ranking.
    ranked = {q["id"]: [(d.metadata["source"], d.metadata["section"], 1 - dist)
                        for d, dist in store.similarity_search_with_score(q["question"], k=max(K_VALUES))]
              for q in questions}
    in_scope = [q for q in questions if q["type"] != "out_of_scope"]

    print(f"{'top_k':>5} {'hit':>7} {'recall':>7} {'MRR':>6} {'precision':>9}   (min_similarity 0.5)")
    for k in K_VALUES:
        m = [retrieval_metrics([(s, sec) for s, sec, sim in ranked[q["id"]][:k] if sim >= 0.5], q["gold"])
             for q in in_scope]
        print(f"{k:>5} {mean(x['hit'] for x in m):>7.1%} {mean(x['recall'] for x in m):>7.1%} "
              f"{mean(x['rr'] for x in m):>6.3f} {mean(x['precision'] for x in m):>9.1%}")

    print("\nQuestions whose gold sections are not all in the top 8:")
    for q in in_scope:
        m = retrieval_metrics([(s, sec) for s, sec, _ in ranked[q["id"]]], q["gold"])
        if m["recall"] < 1:
            print(f"  {q['id']} {q['question']}")

    print("\nRank of the last gold section needed (for questions with more than one gold section):")
    for q in in_scope:
        if len(q["gold"]) > 1:
            ranks = []
            for prefix, needle in q["gold"]:
                r = next((i for i, (s, sec, _) in enumerate(ranked[q["id"]], 1)
                          if s.startswith(prefix) and needle.lower() in sec.lower()), None)
                ranks.append(r)
            print(f"  {q['id']} gold ranks {ranks}")

    top1_in = [ranked[q["id"]][0][2] for q in in_scope]
    top1_out = {q["question"]: ranked[q["id"]][0][2] for q in questions if q["type"] == "out_of_scope"}
    print(f"\nTop-1 similarity, answerable questions: min {min(top1_in):.3f}, mean {mean(top1_in):.3f}")
    for question, sim in top1_out.items():
        print(f"Top-1 similarity, out of scope: {sim:.3f}  {question}")


if __name__ == "__main__":
    main()
