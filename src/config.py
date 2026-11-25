"""Central settings for the dental FAQ chatbot."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DIR = ROOT_DIR / "chroma_db"

COLLECTION_NAME = "dental_faq"

# Embeddings: nomic-embed-text run locally with sentence-transformers (no Ollama).
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# torch's Apple GPU (MPS) backend crashes on macOS 14.5, so the model runs on CPU.
EMBEDDING_DEVICE = "cpu"

# Chunking: each "##" section is one FAQ answer; long sections are split further.
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 150  # characters

# Retrieval. k=5 chosen in evaluation: recall 96.3% -> 98.8% vs k=4, no gain beyond 5 (see evals/REPORT.md).
TOP_K = 5
# Chunks below this cosine similarity are dropped, so off-topic questions get no context
# and the chatbot says it doesn't know instead of guessing.
MIN_SIMILARITY = 0.5

# LLM (OpenAI). The API key is read from OPENAI_API_KEY in .env.
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0  # deterministic answers that stick to the context
# Which system prompt in src/rag_chain.py SYSTEM_PROMPTS to use (older versions are kept for evaluation).
PROMPT_VERSION = "v3"
