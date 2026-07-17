"""
fetch_arxiv.py -- Ingestion stage.

Queries the arXiv API for papers matching a category/search query,
downloads the PDFs, and uploads them (plus a metadata JSON sidecar) to
an S3 "raw" landing zone, partitioned by ingestion date. This is the
landing zone that the S3 event notification watches -- when a new PDF
lands here, it triggers the Lambda function (lambda_handler.py) that
runs the parse -> chunk -> embed -> upsert pipeline.

S3 layout produced by this script:

    s3://<bucket>/raw/run_date=YYYY-MM-DD/<arxiv_id>.pdf
    s3://<bucket>/raw/run_date=YYYY-MM-DD/<arxiv_id>.metadata.json

The metadata sidecar lets the downstream Lambda attach rich metadata
(authors, category, published date) to vector payloads without having
to re-query the arXiv API at processing time.

Usage:
    python fetch_arxiv.py --query "retrieval augmented generation" \
        --category cs.CL --max-results 25 --bucket my-bucket
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import arxiv
import boto3

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    category: str
    published_date: str  # ISO format YYYY-MM-DD
    summary: str
    pdf_url: str


def search_arxiv(
    query: str,
    category: str | None = None,
    max_results: int = 25,
) -> list[arxiv.Result]:
    """Search arXiv for papers matching a query, optionally restricted
    to a category (e.g. 'cs.CL', 'cs.LG', 'cs.CV')."""
    search_query = f"cat:{category} AND {query}" if category else query

    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(client.results(search))
    logger.info("Found %d papers for query: %s", len(results), search_query)
    return results


def _arxiv_id_from_entry(entry_id: str) -> str:
    """Extract a clean arXiv ID (e.g. '2401.12345') from the full entry URL."""
    return entry_id.split("/")[-1].split("v")[0]


def download_and_upload(
    results: list[arxiv.Result],
    bucket: str,
    s3_prefix: str = "raw",
    download_dir: str = "data/raw",
    s3_client=None,
) -> list[PaperMetadata]:
    """Download each paper's PDF and upload it + a metadata sidecar to S3.

    Returns the list of PaperMetadata objects that were successfully
    uploaded (useful for logging / downstream summary).
    """
    s3 = s3_client or boto3.client("s3")
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    uploaded: list[PaperMetadata] = []

    for result in results:
        arxiv_id = _arxiv_id_from_entry(result.entry_id)
        local_pdf_path = download_path / f"{arxiv_id}.pdf"

        try:
            import requests as _requests
            pdf_response = _requests.get(result.pdf_url, timeout=30)
            pdf_response.raise_for_status()
            with open(local_pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_response.content)
        except Exception as e:
            logger.warning("Failed to download %s: %s", arxiv_id, e)
            continue

        metadata = PaperMetadata(
            arxiv_id=arxiv_id,
            title=result.title.strip(),
            authors=[a.name for a in result.authors],
            category=result.primary_category,
            published_date=result.published.strftime("%Y-%m-%d"),
            summary=result.summary.strip().replace("\n", " "),
            pdf_url=result.pdf_url,
        )

        pdf_key = f"{s3_prefix}/run_date={run_date}/{arxiv_id}.pdf"
        metadata_key = f"{s3_prefix}/run_date={run_date}/{arxiv_id}.metadata.json"

        s3.upload_file(str(local_pdf_path), bucket, pdf_key)
        s3.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=json.dumps(asdict(metadata), indent=2),
            ContentType="application/json",
        )

        logger.info("Uploaded s3://%s/%s", bucket, pdf_key)
        uploaded.append(metadata)

    return uploaded


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fetch arXiv papers into an S3 landing zone.")
    parser.add_argument("--query", required=True, help="Search query, e.g. 'retrieval augmented generation'")
    parser.add_argument("--category", default=None, help="arXiv category filter, e.g. cs.CL, cs.LG, cs.CV")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--bucket", required=True, help="S3 bucket name (raw landing zone)")
    parser.add_argument("--s3-prefix", default="raw")
    args = parser.parse_args()

    results = search_arxiv(args.query, category=args.category, max_results=args.max_results)
    uploaded = download_and_upload(results, bucket=args.bucket, s3_prefix=args.s3_prefix)

    print(f"\nIngested {len(uploaded)} papers into s3://{args.bucket}/{args.s3_prefix}/")
    for p in uploaded:
        print(f"  - [{p.arxiv_id}] {p.title} ({p.category}, {p.published_date})")


if __name__ == "__main__":
    main()
