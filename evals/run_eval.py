"""Evaluate the dental FAQ chatbot on a predefined question set.

Measures:
- Retrieval relevance (answerable questions): Hit@k, Recall@k, MRR, and Precision@k against the
  expected ("gold") sections of each question.
- Answer accuracy: LLM-judge correctness (1-5) plus a count of incorrect claims.
- Faithfulness: answers with no claims unsupported by the retrieved context.
- Response completeness: share of the question's reference facts covered by the answer.
- Refusal behavior: out-of-scope questions must be refused; answerable ones must not be.
- Leaks: citations, file names, or mentions of sources in the user-facing answer.

Examples (run from the project root):
    .venv/bin/python -m evals.run_eval --set dev --tag baseline
    .venv/bin/python -m evals.run_eval --set dev --tag k6 --top-k 6 --retrieval-only
"""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from evals.judge import JUDGE_MODEL, Judge
from scripts.test_rag import find_leaks
from src.config import CHUNKS_FILE, LLM_MODEL, MIN_SIMILARITY, PROMPT_VERSION, TOP_K
from src.rag_chain import NO_ANSWER, DentalRAG

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"


def eval_fingerprint(set_name: str) -> str:
    """Hash of everything that defines the evaluation criteria: judge, scoring code, and questions.

    Runs are only comparable if their fingerprints match. Chatbot code (src/) is deliberately excluded.
    """
    h = hashlib.sha256()
    for path in [EVAL_DIR / "judge.py", Path(__file__), EVAL_DIR / f"{set_name}_questions.json"]:
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def load_questions(name: str) -> list[dict]:
    questions = json.loads((EVAL_DIR / f"{name}_questions.json").read_text(encoding="utf-8"))
    validate_gold(questions)
    return questions


def validate_gold(questions: list[dict]) -> None:
    """Fail fast if a gold label doesn't match a real section in the knowledge base."""
    sections = {(json.loads(line)["source"][:2], json.loads(line)["section"]) for line in CHUNKS_FILE.open()}
    for q in questions:
        for prefix, needle in q.get("gold", []):
            if not any(p == prefix and needle.lower() in s.lower() for p, s in sections):
                raise SystemExit(f"{q['id']}: gold section '{prefix} / {needle}' not found in the knowledge base")


def is_gold(source: str, section: str, gold: list[list[str]]) -> int | None:
    """Index of the gold item this chunk matches, or None."""
    for i, (prefix, needle) in enumerate(gold):
        if source.startswith(prefix) and needle.lower() in section.lower():
            return i
    return None


def retrieval_metrics(retrieved: list[tuple[str, str]], gold: list[list[str]]) -> dict:
    matches = [is_gold(src, sec, gold) for src, sec in retrieved]
    first = next((rank for rank, m in enumerate(matches, start=1) if m is not None), None)
    return {
        "hit": first is not None,
        "recall": len({m for m in matches if m is not None}) / len(gold),
        "rr": 1 / first if first else 0.0,
        "precision": sum(m is not None for m in matches) / len(matches) if matches else 0.0,
        "n_retrieved": len(matches),
    }


def evaluate_one(rag: DentalRAG, judge: Judge | None, q: dict, hits) -> dict:
    row = {"id": q["id"], "type": q["type"], "question": q["question"]}
    retrieved = [(d.metadata["source"], d.metadata["section"]) for d, _ in hits]
    row["retrieved"] = [f"{src} > {sec} ({s:.3f})" for (src, sec), (_, s) in zip(retrieved, hits)]
    if q["type"] != "out_of_scope":
        row["retrieval"] = retrieval_metrics(retrieved, q["gold"])
    if judge is None:
        return row

    result = rag.answer(q["question"], hits)
    refused = result.answer == NO_ANSWER
    row.update(answer=result.answer, raw_answer=result.raw_answer, refused=refused,
               cited=[f"{s.source} > {s.section}" for s in result.sources], leaks=find_leaks(result.answer))

    if q["type"] == "out_of_scope":
        row["correct"] = refused
        return row

    if refused:
        row.update(correctness=1, completeness=0.0, accurate=False, faithful=True,
                   incorrect_claims=[], unsupported_claims=[], missed_facts=q["facts"])
        return row

    j, unsupported = judge.evaluate(q["question"], q["facts"], result.contexts, result.answer)
    covered = [fc.covered for fc in j.fact_checks][: len(q["facts"])]
    covered += [False] * (len(q["facts"]) - len(covered))
    row.update(
        correctness=j.correctness,
        completeness=sum(covered) / len(q["facts"]),
        accurate=j.correctness >= 4 and not j.incorrect_claims,
        faithful=not unsupported,
        incorrect_claims=j.incorrect_claims,
        unsupported_claims=unsupported,
        n_claims=len(j.claims),
        missed_facts=[f for f, c in zip(q["facts"], covered) if not c],
    )
    return row


def summarize(rows: list[dict], retrieval_only: bool) -> dict:
    in_scope = [r for r in rows if r["type"] != "out_of_scope"]
    out_scope = [r for r in rows if r["type"] == "out_of_scope"]
    s = {
        "retrieval_hit_at_k": mean(r["retrieval"]["hit"] for r in in_scope),
        "retrieval_recall_at_k": mean(r["retrieval"]["recall"] for r in in_scope),
        "retrieval_mrr": mean(r["retrieval"]["rr"] for r in in_scope),
        "retrieval_precision_at_k": mean(r["retrieval"]["precision"] for r in in_scope),
        "avg_chunks_retrieved": mean(r["retrieval"]["n_retrieved"] for r in in_scope),
    }
    if retrieval_only:
        return s
    s.update(
        answer_accuracy=mean(r["accurate"] for r in in_scope),
        answer_correctness_avg=mean(r["correctness"] for r in in_scope),
        faithfulness=mean(r["faithful"] for r in in_scope),
        completeness=mean(r["completeness"] for r in in_scope),
        false_refusal_rate=mean(r["refused"] for r in in_scope),
        correct_refusal_rate=mean(r["correct"] for r in out_scope) if out_scope else None,
        leak_rate=mean(bool(r["leaks"]) for r in rows),
    )
    return s


def print_summary(s: dict) -> None:
    labels = {
        "retrieval_hit_at_k": "Retrieval Hit@k", "retrieval_recall_at_k": "Retrieval Recall@k",
        "retrieval_mrr": "Retrieval MRR", "retrieval_precision_at_k": "Retrieval Precision@k",
        "avg_chunks_retrieved": "Avg chunks retrieved", "answer_accuracy": "Answer accuracy",
        "answer_correctness_avg": "Correctness (1-5)", "faithfulness": "Faithfulness",
        "completeness": "Completeness", "false_refusal_rate": "False refusals",
        "correct_refusal_rate": "Correct refusals (out of scope)", "leak_rate": "Leaks",
    }
    for key, label in labels.items():
        if key in s and s[key] is not None:
            v = s[key]
            shown = f"{v:.2f}" if key in {"answer_correctness_avg", "avg_chunks_retrieved", "retrieval_mrr"} else f"{v:.1%}"
            print(f"  {label:<32} {shown}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test"], required=True)
    ap.add_argument("--tag", required=True, help="name for this run, used in the results file name")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--min-sim", type=float, default=MIN_SIMILARITY)
    ap.add_argument("--prompt", default=PROMPT_VERSION)
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM; score retrieval only")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    questions = load_questions(args.set)
    rag = DentalRAG(top_k=args.top_k, min_similarity=args.min_sim, prompt_version=args.prompt)
    judge = None if args.retrieval_only else Judge()

    start = time.time()
    # Embedding/retrieval runs sequentially; only the OpenAI calls run in parallel.
    all_hits = [rag.retrieve(q["question"]) for q in questions]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda qh: evaluate_one(rag, judge, *qh), zip(questions, all_hits)))

    summary = summarize(rows, args.retrieval_only)
    config = {"set": args.set, "tag": args.tag, "top_k": args.top_k, "min_similarity": args.min_sim,
              "prompt_version": args.prompt, "llm": LLM_MODEL,
              "judge": None if args.retrieval_only else JUDGE_MODEL, "n_questions": len(questions),
              "eval_fingerprint": eval_fingerprint(args.set),
              "run_at": datetime.now().isoformat(timespec="seconds")}

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}__{args.set}.json"
    if out.exists():
        # Never overwrite a previous run: keep a timestamped copy in the archive first.
        ARCHIVE_DIR.mkdir(exist_ok=True)
        stamp = datetime.fromtimestamp(out.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
        backup = ARCHIVE_DIR / f"{out.stem}.{stamp}.json"
        shutil.copy2(out, backup)
        print(f"Archived previous {out.name} as {backup.relative_to(EVAL_DIR.parent)}")
    out.write_text(json.dumps({"config": config, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False))

    print(f"\n{args.tag} on {args.set} ({len(questions)} questions, top_k={args.top_k}, "
          f"min_sim={args.min_sim}, prompt={args.prompt}) in {time.time() - start:.0f}s")
    print_summary(summary)
    print(f"Eval fingerprint: {config['eval_fingerprint']}")
    print(f"Saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
