"""Check that ChromaDB retrieval returns the right source document for typical questions.

Run from the project root:
    .venv/bin/python -m scripts.test_retrieval
"""
from src.config import TOP_K
from src.vectorstore import get_vectorstore

# (question, expected source file)
CASES = [
    ("What are the first signs of a cavity?", "01_cavities_caries.md"),
    ("Why do my gums bleed when I brush?", "02_gum_disease_periodontitis.md"),
    ("Does a root canal hurt?", "03_root_canal.md"),
    ("How long does a dental implant last?", "04_dental_implants.md"),
    ("What should I do if my tooth gets knocked out?", "05_toothache_dental_emergencies.md"),
    ("What is a canker sore and is it contagious?", "06_oral_mucosa_mouth_sores.md"),
    ("When should my child first go to the dentist?", "07_pediatric_dental_care.md"),
    ("How does högkostnadsskyddet work?", "08_swedish_dental_subsidy_system.md"),
    ("I am 70 years old, what is tiotandvård?", "08_swedish_dental_subsidy_system.md"),
    ("How much is the tandvårdsbidrag for a 25 year old?", "08_swedish_dental_subsidy_system.md"),
    ("Can I buy strong whitening gel in a shop in Sweden?", "09_teeth_whitening.md"),
    ("Are braces free for kids in Sweden?", "10_orthodontics_braces.md"),
    ("Why do my teeth hurt when I drink cold water?", "11_sensitive_teeth_erosion.md"),
    ("What is dry socket after removing a tooth?", "12_wisdom_teeth_extraction.md"),
    ("What is the 2-2-2 method for brushing?", "13_oral_hygiene_prevention.md"),
    ("Is snus bad for my gums?", "14_lifestyle_snus_bad_breath_grinding.md"),
    ("How do I clean my dentures?", "15_crowns_bridges_dentures.md"),
]


def main() -> None:
    store = get_vectorstore()
    hits_at_1 = hits_at_k = 0
    reciprocal_ranks = 0.0

    for question, expected in CASES:
        results = store.similarity_search_with_score(question, k=TOP_K)
        sources = [doc.metadata["source"] for doc, _ in results]
        rank = sources.index(expected) + 1 if expected in sources else None

        hits_at_1 += rank == 1
        hits_at_k += rank is not None
        reciprocal_ranks += 1 / rank if rank else 0

        top_doc, top_distance = results[0]
        mark = "OK  " if rank == 1 else ("~   " if rank else "MISS")
        print(f"{mark} {question}")
        print(f"     top: {top_doc.metadata['source']} > {top_doc.metadata['section']} "
              f"(similarity {1 - top_distance:.3f})")
        if rank != 1:
            print(f"     expected {expected}, found at rank {rank}")

    n = len(CASES)
    print(f"\nHit@1: {hits_at_1}/{n} ({hits_at_1 / n:.0%})   "
          f"Hit@{TOP_K}: {hits_at_k}/{n} ({hits_at_k / n:.0%})   MRR: {reciprocal_ranks / n:.3f}")


if __name__ == "__main__":
    main()
