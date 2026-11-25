"""Run the RAG chain on in-scope, partly-covered, and out-of-scope questions.

Prints the user-facing answer, checks that it contains no citations or source references, and shows
the internal source metadata used for verification.

Run from the project root:
    .venv/bin/python -m scripts.test_rag
"""
import re

from src.rag_chain import NO_ANSWER, DentalRAG

QUESTIONS = [
    # Answer is in the knowledge base
    "I'm 70. How much do I pay for a crown on a front tooth?",
    "How much tandvårdsbidrag does a 25 year old get?",
    "Until what age is dental care free in Sweden?",
    "My face is swollen and I have a fever after a toothache. What should I do?",
    # Only partly covered
    "How much does a filling cost at Folktandvården in Malmö in 2027?",
    # Not covered: should refuse, not invent
    "What is the best dentist in Stockholm?",
    "Can Botox help with teeth grinding?",
    "What is the capital of France?",
]

# Things that must never reach the user.
LEAK_PATTERNS = {
    "citation marker": r"\[\d",
    "file name": r"\.md\b|\b\d{2}_[a-z_]+",
    "mentions context/sources": r"\b(context|passage|document|source)s?\b|knowledge base",
}


def find_leaks(answer: str) -> list[str]:
    return [name for name, pattern in LEAK_PATTERNS.items() if re.search(pattern, answer, re.IGNORECASE)]


def main() -> None:
    rag = DentalRAG()
    problems = 0
    for q in QUESTIONS:
        r = rag.ask(q)
        leaks = find_leaks(r.answer)
        # An answer (not a refusal) with no cited passage can't be traced back to the knowledge base.
        ungrounded = r.answer != NO_ANSWER and not r.sources
        problems += bool(leaks) + ungrounded

        print("=" * 90)
        print("Q:", q)
        print("User sees:", r.answer)
        print("Internal sources:", "; ".join(f"[{s.number}] {s.source} > {s.section}" for s in r.sources) or "none")
        if leaks:
            print("  LEAK:", ", ".join(leaks))
        if ungrounded:
            print("  UNGROUNDED: answer cites no retrieved passage")

    print("=" * 90)
    print(f"{len(QUESTIONS)} questions, {problems} problems")


if __name__ == "__main__":
    main()
