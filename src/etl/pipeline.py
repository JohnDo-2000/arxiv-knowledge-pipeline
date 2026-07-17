"""
pipeline.py -- Orchestrates the full parse -> chunk -> embed -> upsert flow
for a single document. Used by both the local demo script and the AWS
Lambda handler, so the core logic is defined once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from etl.chunk import chunk_text
from etl.embed import embed_texts
from etl.parse import parse_pdf
from vectorstore.qdrant_client import VectorStore, _date_to_epoch

logger = logging.getLogger(__name__)


def process_document(
    pdf_path: str | Path,
    document_metadata: dict[str, Any],
    vector_store: VectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """Run one document through the full pipeline and upsert its chunks.

    Args:
        pdf_path: Path to the PDF file to process.
        document_metadata: Document-level metadata to attach to every
            chunk (arxiv_id, title, authors, category, published_date).
            Expects "published_date" as an ISO string (YYYY-MM-DD); this
            function derives "published_epoch" automatically for Qdrant
            range filtering.
        vector_store: An initialized VectorStore (collection should
            already exist -- call ensure_collection() once at startup,
            not per document).
        chunk_size / chunk_overlap: Passed through to chunk_text.

    Returns:
        Number of chunks upserted for this document.
    """
    pdf_path = Path(pdf_path)
    logger.info("Processing document: %s", pdf_path.name)

    # 1. Parse + clean
    parsed = parse_pdf(pdf_path)
    if not parsed.full_text.strip():
        logger.warning("No extractable text in %s, skipping.", pdf_path.name)
        return 0

    # 2. Chunk
    metadata = dict(document_metadata)
    if "published_date" in metadata:
        metadata["published_epoch"] = _date_to_epoch(metadata["published_date"])

    chunks = chunk_text(
        parsed.full_text,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        logger.warning("No chunks produced for %s, skipping.", pdf_path.name)
        return 0

    # 3. Embed
    chunk_texts = [c.text for c in chunks]
    vectors = embed_texts(chunk_texts)

    # 4. Upsert
    payloads = [c.to_payload() for c in chunks]
    n = vector_store.upsert_chunks(vectors.tolist(), payloads)

    logger.info("Finished %s: %d chunks upserted", pdf_path.name, n)
    return n
