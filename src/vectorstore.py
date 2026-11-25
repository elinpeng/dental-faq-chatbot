"""Access to the persistent ChromaDB collection."""
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.embeddings import NomicEmbeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=NomicEmbeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
