"""Smoke test: load nomic-embed-text locally and check that retrieval similarity makes sense."""
import time

from sentence_transformers import SentenceTransformer

MODEL = "nomic-ai/nomic-embed-text-v1.5"

start = time.time()
print(f"Loading {MODEL} ...", flush=True)
# CPU: torch's Apple GPU (MPS) backend crashes on this macOS version, and CPU is fast enough for this corpus.
model = SentenceTransformer(MODEL, trust_remote_code=True, device="cpu")
print(f"Loaded in {time.time() - start:.1f}s on device: {model.device}", flush=True)

# nomic-embed-text requires task prefixes: "search_document: " for chunks, "search_query: " for questions.
docs = [
    "Högkostnadsskyddet: you pay the first 3,000 SEK, then 50% up to 15,000 SEK, and 15% above that.",
    "Dry socket happens when the blood clot falls out after a tooth extraction.",
    "Whitening products sold in shops may contain at most 0.1% hydrogen peroxide.",
]
query = "How much do I pay for expensive dental treatment in Sweden?"

doc_vecs = model.encode([f"search_document: {d}" for d in docs], normalize_embeddings=True)
query_vec = model.encode([f"search_query: {query}"], normalize_embeddings=True)[0]

print("Embedding dimension:", doc_vecs.shape[1])
print("Query:", query)
for doc, score in sorted(zip(docs, doc_vecs @ query_vec), key=lambda x: -x[1]):
    print(f"  {score:.3f}  {doc[:70]}")
print("OK", flush=True)
