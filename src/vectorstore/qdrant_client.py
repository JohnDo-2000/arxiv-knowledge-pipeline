"""
qdrant_client.py -- Vector store interface.

Wraps the Qdrant client for this pipeline: collection creation, upserting
chunk embeddings with metadata payloads, and metadata-filtered semantic
search (e.g. "find similar chunks, but only from cs.CL papers published
after 2024-01-01").

Qdrant runs as a local Docker container during development:

    docker run -p 6333:6333 -p 6334:6334 \
        -v "$(pwd)/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant

See docker-compose.yml for the equivalent Compose definition.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "arxiv_papers"


class VectorStore:
    """Thin wrapper around QdrantClient scoped to this pipeline's collection."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_dim: int = 384,
    ):
        # Passing url=":memory:" runs Qdrant fully in-process with no
        # server required -- handy for unit tests / quick local checks.
        # Real usage (and the Lambda handler) should point at the actual
        # Qdrant server, e.g. "http://localhost:6333" locally or your
        # deployed Qdrant host's URL in production.
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist (or recreate it).

        Uses Cosine distance, which expects normalized vectors -- matches
        the `normalize=True` default in embed.py.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            logger.info("Recreating collection: %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(
                "Creating collection '%s' (dim=%d, distance=Cosine)",
                self.collection_name,
                self.embedding_dim,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
        else:
            logger.info("Collection '%s' already exists", self.collection_name)

    def upsert_chunks(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> int:
        """Upsert a batch of (vector, metadata payload) pairs.

        Each payload should contain the chunk text plus document-level
        metadata (arxiv_id, title, authors, category, published_date,
        chunk_index) so it can be filtered on at query time.

        Returns the number of points upserted.
        """
        if len(vectors) != len(payloads):
            raise ValueError(
                f"vectors ({len(vectors)}) and payloads ({len(payloads)}) length mismatch"
            )

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted %d points into '%s'", len(points), self.collection_name)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        category: str | None = None,
        author: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search with optional metadata filters.

        Args:
            query_vector: Embedding of the search query.
            top_k: Number of results to return.
            category: Filter by exact arXiv category (e.g. "cs.CL").
            author: Filter to chunks whose author list contains this name.
            published_after: ISO date string (e.g. "2024-01-01"); filters
                to papers published on or after this date.
            published_before: ISO date string; filters to papers published
                on or before this date.

        Returns:
            List of dicts with score and payload (chunk text + metadata).
        """
        must_conditions: list[FieldCondition] = []

        if category:
            must_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if author:
            must_conditions.append(
                FieldCondition(key="authors", match=MatchValue(value=author))
            )
        if published_after or published_before:
            range_kwargs: dict[str, Any] = {}
            if published_after:
                range_kwargs["gte"] = _date_to_epoch(published_after)
            if published_before:
                range_kwargs["lte"] = _date_to_epoch(published_before)
            must_conditions.append(
                FieldCondition(key="published_epoch", range=Range(**range_kwargs))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        return [{"score": point.score, "payload": point.payload} for point in results.points]

    def count(self) -> int:
        """Return the number of points currently in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0


def _date_to_epoch(date_str: str) -> float:
    """Convert an ISO date string (YYYY-MM-DD) to a Unix epoch timestamp,
    for use in Qdrant Range filters on the `published_epoch` field."""
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    store = VectorStore()
    try:
        store.ensure_collection()
        print(f"Collection '{store.collection_name}' ready. Point count: {store.count()}")
    except Exception as e:
        print(f"Could not connect to Qdrant at http://localhost:6333 -- is it running?")
        print(f"Start it with: docker compose up -d")
        print(f"Error: {e}")
        sys.exit(1)
