"""
parse.py -- Document parsing and cleaning stage.

Uses unstructured.io to partition a PDF into typed elements (Title,
NarrativeText, ListItem, Header, Footer, etc.), then filters and cleans
those elements into a single normalized text block ready for chunking.

IMPORTANT SETUP NOTE (read before running):
---------------------------------------------
unstructured's PDF partitioner depends on a spaCy English model
(en_core_web_sm) to classify text spans (title vs. narrative vs. list,
etc). By default it tries to download this model from a GitHub release
URL the FIRST time you run it. This has two consequences:

  1. First run requires internet access to GitHub release assets.
     If you're behind a restrictive proxy/firewall, this download can
     fail with an HTTP 403/timeout.
  2. If you containerize this for AWS Lambda, you do NOT want this
     download happening inside the Lambda execution environment
     (slow cold starts, possible network restrictions, non-determinism).
     Instead, bake the model into the Docker image at BUILD time:

         RUN python -m spacy download en_core_web_sm

     See infra/Dockerfile.lambda for where this is handled.

Run `python -m spacy download en_core_web_sm` once locally before using
this module for the first time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

# Element types we treat as noise / boilerplate and drop before chunking.
# arXiv PDFs commonly have running headers (e.g. "arXiv:2401.12345v1 [cs.LG] ...")
# and page-number footers that we don't want polluting semantic chunks.
NOISE_ELEMENT_TYPES = {"Header", "Footer", "PageBreak"}

# Regex patterns that catch boilerplate unstructured sometimes misclassifies
# as NarrativeText (e.g. a running arXiv header that lands in the body text).
NOISE_LINE_PATTERNS = [
    re.compile(r"^arXiv:\d{4}\.\d{4,5}v\d+\s*\[.+?\]\s*\d{1,2}\s+\w+\s+\d{4}$"),
    re.compile(r"^\s*\d+\s*$"),  # bare page-number lines
]


@dataclass
class ParsedDocument:
    """Result of parsing a single PDF: cleaned full text plus structure."""

    source_path: str
    full_text: str
    titles: list[str]
    element_count: int
    raw_elements: list[Element]


def _is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(pattern.match(stripped) for pattern in NOISE_LINE_PATTERNS)


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip stray markdown-ish artifacts that
    sometimes survive PDF extraction (e.g. repeated dashes from tables,
    stray pipe characters from malformed table parsing)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[-=_]{3,}$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_pdf(file_path: str | Path, strategy: str = "fast") -> ParsedDocument:
    """Parse a PDF into a cleaned ParsedDocument.

    Args:
        file_path: Path to the PDF file.
        strategy: unstructured partitioning strategy. "fast" uses pdfminer
            text extraction directly (good for born-digital PDFs like
            arXiv papers). Use "hi_res" if you need layout-model-based
            parsing for scanned documents or complex tables -- this is
            slower and pulls in a heavier model.

    Returns:
        ParsedDocument with cleaned, concatenated text and basic structure.
    """
    file_path = Path(file_path)
    logger.info("Parsing PDF: %s (strategy=%s)", file_path.name, strategy)

    elements = partition_pdf(filename=str(file_path), strategy=strategy)

    titles: list[str] = []
    kept_text_blocks: list[str] = []

    for element in elements:
        element_type = type(element).__name__
        text = str(element)

        if element_type in NOISE_ELEMENT_TYPES:
            continue
        if _is_noise_text(text):
            continue

        if element_type == "Title":
            titles.append(text.strip())

        kept_text_blocks.append(text.strip())

    full_text = _clean_text("\n\n".join(kept_text_blocks))

    logger.info(
        "Parsed %s: %d raw elements -> %d kept text blocks (%d chars)",
        file_path.name,
        len(elements),
        len(kept_text_blocks),
        len(full_text),
    )

    return ParsedDocument(
        source_path=str(file_path),
        full_text=full_text,
        titles=titles,
        element_count=len(elements),
        raw_elements=list(elements),
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python parse.py <path_to_pdf>")
        sys.exit(1)

    doc = parse_pdf(sys.argv[1])
    print(f"\n--- Titles found ---")
    for t in doc.titles:
        print(f"  - {t}")
    print(f"\n--- First 1000 chars of cleaned text ---")
    print(doc.full_text[:1000])
