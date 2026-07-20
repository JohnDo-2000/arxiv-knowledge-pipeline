from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import unquote_plus
from typing import Any

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "arxiv_papers")

s3 = boto3.client("s3")
_vector_store = None


def _get_vector_store():
    from vectorstore.qdrant_client import VectorStore
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(url=QDRANT_URL, collection_name=COLLECTION_NAME)
        _vector_store.ensure_collection()
    return _vector_store


def _load_metadata_sidecar(bucket, pdf_key):
    metadata_key = pdf_key.replace(".pdf", ".metadata.json")
    try:
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        return json.loads(response["Body"].read())
    except Exception:
        arxiv_id = Path(pdf_key).stem
        return {"arxiv_id": arxiv_id, "title": arxiv_id}


def handler(event, context):
    from etl.pipeline import process_document
    vector_store = _get_vector_store()
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        pdf_key = unquote_plus(record["s3"]["object"]["key"])
        if not pdf_key.endswith(".pdf"):
            continue
        logger.info("Processing s3://%s/%s", bucket, pdf_key)
        local_pdf_path = Path("/tmp") / Path(pdf_key).name
        s3.download_file(bucket, pdf_key, str(local_pdf_path))
        metadata = _load_metadata_sidecar(bucket, pdf_key)
        try:
            n_chunks = process_document(pdf_path=local_pdf_path, document_metadata=metadata, vector_store=vector_store)
            results.append({"key": pdf_key, "status": "success", "chunks_upserted": n_chunks})
        except Exception as e:
            logger.exception("Failed to process %s", pdf_key)
            results.append({"key": pdf_key, "status": "error", "error": str(e)})
        finally:
            local_pdf_path.unlink(missing_ok=True)
    return {"statusCode": 200, "results": results}
