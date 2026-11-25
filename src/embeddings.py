"""LangChain wrapper for nomic-embed-text running locally with sentence-transformers."""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_DEVICE, EMBEDDING_MODEL

# nomic-embed-text is trained with task prefixes; leaving them out lowers retrieval quality.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=EMBEDDING_DEVICE)


class NomicEmbeddings(Embeddings):
    """Embeds chunks and questions with nomic-embed-text, normalized for cosine similarity."""

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = _load_model().encode(
            [DOCUMENT_PREFIX + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = _load_model().encode([QUERY_PREFIX + text], normalize_embeddings=True)[0]
        return vector.tolist()
