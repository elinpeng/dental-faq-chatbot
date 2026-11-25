"""Clean and chunk the dental documents, embed them with nomic-embed-text, and store them in ChromaDB.

Run from the project root:
    .venv/bin/python -m src.ingest
"""
import json
import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE, PROCESSED_DIR, RAW_DIR, ROOT_DIR
from src.vectorstore import get_vectorstore


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace, and drop markdown emphasis that adds noise to embeddings."""
    text = unicodedata.normalize("NFC", text)  # keeps å/ä/ö consistent
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** -> bold
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_chunk() -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=True,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n- ", "\n", ". ", " "],
    )

    files = sorted(RAW_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {RAW_DIR}")

    chunks: list[Document] = []
    for path in files:
        text = clean_text(path.read_text(encoding="utf-8"))
        if not text:
            raise SystemExit(f"{path.name} is empty. Restore it before ingesting.")
        sections = [s for s in header_splitter.split_text(text) if s.page_content.strip()]
        for section_idx, section in enumerate(sections):
            title = section.metadata.get("title", path.stem)
            heading = section.metadata.get("section", "")
            # Put the heading in each chunk so the embedding knows what question the passage answers.
            context = f"{title} > {heading}" if heading else title
            # The header splitter joins lines with "  \n" (markdown line breaks); strip those spaces.
            body = re.sub(r"[ \t]+\n", "\n", section.page_content)
            for part_idx, part in enumerate(size_splitter.split_text(body)):
                chunks.append(
                    Document(
                        page_content=f"{context}\n{part}",
                        metadata={
                            "chunk_id": f"{path.stem}::{section_idx:02d}::{part_idx}",
                            "source": path.name,
                            "title": title,
                            "section": heading,
                        },
                    )
                )
    return chunks


def save_chunks(chunks: list[Document]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, **c.metadata}, ensure_ascii=False) + "\n")


def build_index(chunks: list[Document]) -> int:
    store = get_vectorstore()
    store.delete_collection()  # rebuild from scratch so edited or removed text doesn't linger
    store = get_vectorstore()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return store._collection.count()


def main() -> None:
    chunks = load_and_chunk()
    save_chunks(chunks)
    lengths = [len(c.page_content) for c in chunks]
    sources = {c.metadata["source"] for c in chunks}
    print(f"Chunked {len(sources)} documents into {len(chunks)} chunks "
          f"(chars: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)})")
    print(f"Saved cleaned chunks to {CHUNKS_FILE.relative_to(ROOT_DIR)}")
    count = build_index(chunks)
    print(f"Stored {count} embeddings in ChromaDB at {CHUNKS_FILE.parents[2].name}/chroma_db")


if __name__ == "__main__":
    main()
