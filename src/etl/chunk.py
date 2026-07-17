"""
chunk.py -- Semantic chunking stage.

Splits cleaned document text into retrieval-sized chunks using LangChain's
RecursiveCharacterTextSplitter, which tries to break on natural boundaries
(paragraph, then sentence, then word) before falling back to a hard
character cut. This avoids slicing a sentence in half mid-thought, which
fixed-offset chunking does and which hurts retrieval quality.

Each chunk carries source metadata so it can be filtered later in the
vector store (arxiv_id, title, chunk_index, etc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tuned for a sentence-transformer embedding model with a 256-384 token
# context window (e.g. all-MiniLM-L6-v2 caps at 256 tokens). ~1000 chars
# is a safe approximation of ~200-250 tokens for English text.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Order matters: try to split on paragraph breaks first, then lines,
# then sentences, then words, then characters as a last resort.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single semantic chunk of text plus its metadata payload."""

    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Metadata dict suitable for storing alongside the vector in Qdrant."""
        return {"text": self.text, "chunk_index": self.chunk_index, **self.metadata}


def chunk_text(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split text into semantic chunks with attached metadata.

    Args:
        text: Cleaned document text (output of parse.py).
        metadata: Document-level metadata to attach to every chunk
            (e.g. arxiv_id, title, authors, published_date, category).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks, in characters,
            so context isn't lost at chunk boundaries.

    Returns:
        List of Chunk objects, each with chunk_index and shared document
        metadata plus a per-chunk char_count.
    """
    metadata = metadata or {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )

    raw_chunks = splitter.split_text(text)

    chunks: list[Chunk] = []
    for i, raw_chunk in enumerate(raw_chunks):
        chunk_metadata = {**metadata, "char_count": len(raw_chunk)}
        chunks.append(Chunk(text=raw_chunk, chunk_index=i, metadata=chunk_metadata))

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python chunk.py <path_to_text_file>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        sample_text = f.read()

    sample_metadata = {
        "arxiv_id": "2401.99999",
        "title": "Efficient Retrieval-Augmented Generation via Hierarchical Semantic Chunking",
        "authors": ["Jane A. Researcher", "Wei Zhang", "Carlos M. Ortega"],
        "category": "cs.LG",
        "published_date": "2024-01-15",
    }

    result_chunks = chunk_text(sample_text, metadata=sample_metadata)

    print(f"Produced {len(result_chunks)} chunks\n")
    for c in result_chunks:
        print(f"--- Chunk {c.chunk_index} ({c.metadata['char_count']} chars) ---")
        print(c.text[:200].replace("\n", " "))
        print()
