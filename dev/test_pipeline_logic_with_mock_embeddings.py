"""
DEV-ONLY test script -- validates the chunk -> upsert -> metadata-filtered
search logic end-to-end using a fake embedding function (random vectors
seeded by text hash, so identical text always gets the same vector).

This exists because the sandbox used to build this repo can't reach
huggingface.co to download real model weights. On your machine, use
scripts/run_local_demo.py instead, which uses the REAL embed.py with
real sentence-transformers embeddings.

This script proves the Qdrant collection setup, payload structure, and
metadata filter logic (category / author / date range) all work
correctly, independent of which embedding model produced the vectors.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from etl.chunk import chunk_text  # noqa: E402
from vectorstore.qdrant_client import VectorStore, _date_to_epoch  # noqa: E402

EMBEDDING_DIM = 384


def fake_embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding: hash the text into a seed, then
    generate a reproducible random unit vector. Same text -> same vector,
    different text -> different (effectively random, no real semantics)
    vector. This is ONLY for testing storage/filtering logic, not for
    testing retrieval quality (which needs real embeddings)."""
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=EMBEDDING_DIM)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# Three fake "papers" with distinct metadata, so we can prove filtering works.
FAKE_PAPERS = [
    {
        "text": (
            "Hierarchical semantic chunking improves retrieval precision "
            "in RAG pipelines by respecting topic boundaries. We evaluate "
            "on three QA benchmarks and show consistent gains.\n\n"
            "Future work includes extending this approach to multi-modal "
            "documents and learned boundary detection."
        ),
        "metadata": {
            "arxiv_id": "2401.99999",
            "title": "Efficient RAG via Hierarchical Semantic Chunking",
            "authors": "Jane A. Researcher",
            "category": "cs.LG",
            "published_date": "2024-01-15",
        },
    },
    {
        "text": (
            "We introduce a new benchmark for evaluating dense passage "
            "retrieval systems across 12 languages. Our results show "
            "significant performance gaps between high- and low-resource "
            "languages in multilingual retrieval settings."
        ),
        "metadata": {
            "arxiv_id": "2403.55555",
            "title": "Multilingual Dense Retrieval Benchmark",
            "authors": "Wei Zhang",
            "category": "cs.CL",
            "published_date": "2024-03-22",
        },
    },
    {
        "text": (
            "Vision transformers can be pruned by up to 40% with minimal "
            "accuracy loss using our structured sparsity method. We "
            "demonstrate results on ImageNet classification and COCO "
            "object detection benchmarks."
        ),
        "metadata": {
            "arxiv_id": "2312.77777",
            "title": "Structured Pruning for Vision Transformers",
            "authors": "Carlos M. Ortega",
            "category": "cs.CV",
            "published_date": "2023-12-01",
        },
    },
]


def main() -> None:
    store = VectorStore(url=":memory:", embedding_dim=EMBEDDING_DIM)
    store.ensure_collection()

    all_vectors: list[list[float]] = []
    all_payloads: list[dict] = []

    for paper in FAKE_PAPERS:
        meta = dict(paper["metadata"])
        meta["published_epoch"] = _date_to_epoch(meta["published_date"])

        chunks = chunk_text(paper["text"], metadata=meta, chunk_size=300, chunk_overlap=30)
        print(f"'{meta['title']}' -> {len(chunks)} chunk(s)")

        for chunk in chunks:
            all_vectors.append(fake_embed(chunk.text))
            all_payloads.append(chunk.to_payload())

    n = store.upsert_chunks(all_vectors, all_payloads)
    print(f"\nUpserted {n} chunks. Collection now has {store.count()} points.\n")

    # --- Test 1: unfiltered search ---
    query_vec = fake_embed("retrieval augmented generation chunking")
    results = store.search(query_vec, top_k=3)
    print("=== Test 1: Unfiltered search (top 3) ===")
    for r in results:
        print(f"  score={r['score']:.4f}  category={r['payload']['category']}  "
              f"title={r['payload']['title']}")

    # --- Test 2: filter by category ---
    results = store.search(query_vec, top_k=5, category="cs.CV")
    print("\n=== Test 2: Filtered by category=cs.CV (should only return the pruning paper) ===")
    for r in results:
        print(f"  category={r['payload']['category']}  title={r['payload']['title']}")
    assert all(r["payload"]["category"] == "cs.CV" for r in results), "category filter failed!"

    # --- Test 3: filter by published_after ---
    results = store.search(query_vec, top_k=5, published_after="2024-01-01")
    print("\n=== Test 3: Filtered by published_after=2024-01-01 (should exclude the 2023 paper) ===")
    for r in results:
        print(f"  published_date={r['payload']['published_date']}  title={r['payload']['title']}")
    assert all(
        r["payload"]["published_date"] >= "2024-01-01" for r in results
    ), "date filter failed!"

    # --- Test 4: filter by author ---
    results = store.search(query_vec, top_k=5, author="Wei Zhang")
    print("\n=== Test 4: Filtered by author='Wei Zhang' ===")
    for r in results:
        print(f"  authors={r['payload']['authors']}  title={r['payload']['title']}")
    assert all(r["payload"]["authors"] == "Wei Zhang" for r in results), "author filter failed!"

    print("\nAll metadata filter checks passed.")


if __name__ == "__main__":
    main()
