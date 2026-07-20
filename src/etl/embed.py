"""
embed.py -- Embedding generation stage.

Generates dense vector embeddings for text chunks using a local,
free, open-source HuggingFace sentence-transformers model. No API key
or paid service required -- the model runs on CPU (or GPU if available)
on whatever machine executes this code.

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~80MB model size (small enough to bake into a Lambda container image)
  - Strong quality/speed tradeoff for semantic search use cases
  - Max sequence length ~256 tokens, which is why chunk.py targets
    ~1000 characters (~200-250 tokens) per chunk

NOTE ON FIRST RUN: the first time you instantiate SentenceTransformer
with a given model name, it downloads the model weights from
huggingface.co and caches them locally (~/.cache/huggingface by
default). Subsequent runs are fully offline. For the Lambda container
image, we pre-download the model at BUILD time (see infra/Dockerfile.lambda)
so cold starts never hit the network.
"""

from __future__ import annotations
import os 
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match the model above; used when creating the Qdrant collection

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load (and cache) a sentence-transformers model.

    Cached at module level so repeated calls within the same process
    (e.g. embedding many chunks in a loop) don't reload the model
    from disk every time.
    """
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME", None)
        _model_cache[model_name] = SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            local_files_only=True,
        )
    return _model_cache[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings (e.g. chunk.text values).
        model_name: HuggingFace model identifier.
        batch_size: Batch size for encoding (tune based on available memory).
        normalize: If True, L2-normalize embeddings so cosine similarity
            reduces to a dot product -- standard practice for
            sentence-transformers models and what Qdrant's Cosine
            distance metric expects.

    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    model = get_model(model_name)
    logger.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_texts = [
        "Retrieval-augmented generation depends on chunk quality.",
        "The stock market fell sharply on Tuesday amid rate fears.",
        "Semantic chunking respects topic boundaries in documents.",
    ]

    vectors = embed_texts(sample_texts)
    print(f"Shape: {vectors.shape}")
    print(f"Norm of first vector (should be ~1.0 if normalized): {np.linalg.norm(vectors[0]):.4f}")

    # Sanity check: texts 0 and 2 are semantically related (chunking/RAG),
    # text 1 is unrelated (finance). Cosine similarity should reflect that.
    sim_0_2 = np.dot(vectors[0], vectors[2])
    sim_0_1 = np.dot(vectors[0], vectors[1])
    print(f"Similarity(chunking text, chunking text): {sim_0_2:.4f}")
    print(f"Similarity(chunking text, finance text): {sim_0_1:.4f}")
