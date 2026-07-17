"""
lambda_handler.py -- AWS Lambda entrypoint.

Triggered by an S3 event notification configured on the raw landing
zone bucket (PutObject events, filtered to *.pdf keys, see
infra/deploy_notes.md for the exact event-source-mapping setup).

For each new PDF that lands in S3:
  1. Download it (and its metadata sidecar) into Lambda's /tmp.
  2. Run it through the shared pipeline (parse -> chunk -> embed -> upsert).
  3. Log a structured result to CloudWatch.

Deployment note: this function is packaged as a Lambda CONTAINER IMAGE
(not a zip), because unstructured + sentence-transformers + their
dependencies are too large/native for the zip deployment path, and
because the spaCy model and embedding model need to be baked into the
image at build time (see infra/Dockerfile.lambda) -- fetching them at
runtime would mean every cold start needs network access to
GitHub/HuggingFace, which is slow and a needless failure point.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import boto3

from etl.pipeline import process_document
from vectorstore.qdrant_client import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant connection -- in production this points at wherever you've
# deployed your Qdrant server (e.g. an EC2 instance or Qdrant Cloud),
# passed in via Lambda environment variables. It must NOT be localhost,
# since Lambda execution environments don't have a local Qdrant server.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "arxiv_papers")

s3 = boto3.client("s3")
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    """Lazily initialize the VectorStore once per Lambda execution
    environment (reused across warm invocations)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(url=QDRANT_URL, collection_name=COLLECTION_NAME)
        _vector_store.ensure_collection()
    return _vector_store


def _load_metadata_sidecar(bucket: str, pdf_key: str) -> dict[str, Any]:
    """Load the *.metadata.json sidecar uploaded alongside the PDF by
    fetch_arxiv.py. Falls back to minimal metadata derived from the key
    if the sidecar is missing."""
    metadata_key = pdf_key.replace(".pdf", ".metadata.json")
    try:
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        return json.loads(response["Body"].read())
    except s3.exceptions.NoSuchKey:
        logger.warning("No metadata sidecar found at %s, using minimal metadata.", metadata_key)
        arxiv_id = Path(pdf_key).stem
        return {"arxiv_id": arxiv_id, "title": arxiv_id}


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entrypoint. Processes every S3 object referenced in the event."""
    vector_store = _get_vector_store()
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        pdf_key = record["s3"]["object"]["key"]

        if not pdf_key.endswith(".pdf"):
            logger.info("Skipping non-PDF key: %s", pdf_key)
            continue

        logger.info("Processing s3://%s/%s", bucket, pdf_key)

        local_pdf_path = Path("/tmp") / Path(pdf_key).name
        s3.download_file(bucket, pdf_key, str(local_pdf_path))

        metadata = _load_metadata_sidecar(bucket, pdf_key)

        try:
            n_chunks = process_document(
                pdf_path=local_pdf_path,
                document_metadata=metadata,
                vector_store=vector_store,
            )
            results.append({"key": pdf_key, "status": "success", "chunks_upserted": n_chunks})
        except Exception as e:
            logger.exception("Failed to process %s", pdf_key)
            results.append({"key": pdf_key, "status": "error", "error": str(e)})
        finally:
            local_pdf_path.unlink(missing_ok=True)

    return {"statusCode": 200, "results": results}
