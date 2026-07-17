"""
DEV-ONLY HELPER -- not part of the actual pipeline.

The sandbox this was built in cannot reach arxiv.org (network allowlist),
so this script generates a synthetic but structurally realistic
arXiv-style paper (abstract, sections, headers/footers, references) so we
can develop and test parse.py / chunk.py / embed.py end to end.

On your own machine, fetch_arxiv.py will download REAL papers instead --
this file exists purely so I could test the pipeline in this environment.
You can delete it once you've confirmed the real pipeline works for you.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

OUTPUT_PATH = "data/sample_pdfs/2401.99999_test_paper.pdf"

PAGE_HEADER = "arXiv:2401.99999v1 [cs.LG] 15 Jan 2024"

TITLE = "Efficient Retrieval-Augmented Generation via Hierarchical Semantic Chunking"

AUTHORS = "Jane A. Researcher, Wei Zhang, Carlos M. Ortega"
AFFIL = "Department of Computer Science, Fictional University"

ABSTRACT = (
    "Retrieval-augmented generation (RAG) systems depend critically on the "
    "quality of text chunking strategies used during the indexing phase. "
    "Naive fixed-length chunking frequently splits semantically coherent "
    "passages, degrading downstream retrieval accuracy. In this work, we "
    "propose a hierarchical semantic chunking method that leverages "
    "sentence-level embeddings to detect topic boundaries before "
    "constructing chunks. We evaluate our method on three open-domain "
    "question answering benchmarks and show consistent improvements in "
    "retrieval precision (+6.2 points) and downstream answer accuracy "
    "(+4.8 points) compared to fixed-length baselines."
)

SECTIONS = [
    ("1. Introduction", (
        "Large language models (LLMs) augmented with external retrieval "
        "have become the dominant paradigm for knowledge-intensive NLP "
        "tasks. The retrieval component typically operates over a vector "
        "index built by embedding chunks of source documents. The choice "
        "of chunking strategy therefore has an outsized effect on overall "
        "system quality, yet it remains comparatively understudied "
        "relative to embedding model design.\n\n"
        "Prior work has largely relied on fixed-length character or token "
        "windows, optionally with overlap, to construct chunks. While "
        "simple to implement, this approach ignores document structure "
        "and topic boundaries, often producing chunks that mix unrelated "
        "content or split a single idea across multiple chunks."
    )),
    ("2. Related Work", (
        "Semantic chunking has been explored in the context of "
        "summarization and topic segmentation. Hearst (1997) introduced "
        "TextTiling, a lexical cohesion-based method for detecting "
        "topic shifts in long documents. More recent neural approaches "
        "use sentence embeddings and cosine similarity thresholds to "
        "identify boundaries, an approach we extend in Section 3.\n\n"
        "Within the RAG literature, Lewis et al. (2020) introduced the "
        "original retrieval-augmented generation architecture, "
        "combining a dense retriever with a sequence-to-sequence "
        "generator. Subsequent work has focused primarily on improving "
        "the retriever and generator components in isolation, with "
        "comparatively little attention paid to chunk construction."
    )),
    ("3. Method", (
        "Our hierarchical semantic chunking algorithm proceeds in three "
        "stages. First, we split the input document into sentences using "
        "a rule-based sentence tokenizer. Second, we compute dense "
        "embeddings for each sentence using a pretrained sentence "
        "transformer model. Third, we compute the cosine distance "
        "between consecutive sentence embeddings and insert a chunk "
        "boundary wherever this distance exceeds an adaptive threshold "
        "derived from the local distance distribution.\n\n"
        "This produces variable-length chunks that respect topic "
        "boundaries while remaining within a maximum token budget "
        "suitable for downstream embedding models."
    )),
    ("4. Experiments", (
        "We evaluate on three open-domain QA benchmarks: Natural "
        "Questions, TriviaQA, and HotpotQA. For each benchmark, we "
        "construct a retrieval index using both fixed-length chunking "
        "(512 tokens, 50-token overlap) and our hierarchical semantic "
        "chunking method, holding the embedding model and retriever "
        "constant across conditions.\n\n"
        "We report retrieval precision@5 and downstream answer exact "
        "match (EM) after passing retrieved chunks to a fixed "
        "generator model."
    )),
    ("5. Results", (
        "Across all three benchmarks, hierarchical semantic chunking "
        "improves retrieval precision@5 by an average of 6.2 points "
        "and downstream answer EM by 4.8 points relative to the "
        "fixed-length baseline. We observe the largest gains on "
        "HotpotQA, where multi-hop questions benefit disproportionately "
        "from chunks that preserve complete reasoning steps."
    )),
    ("6. Conclusion", (
        "We have presented a hierarchical semantic chunking method for "
        "retrieval-augmented generation pipelines and demonstrated "
        "consistent improvements over fixed-length baselines across "
        "three QA benchmarks. Future work includes extending this "
        "approach to multi-modal documents and exploring learned, "
        "rather than threshold-based, boundary detection."
    )),
]

REFERENCES = (
    "References\n"
    "[1] Hearst, M. (1997). TextTiling: Segmenting text into "
    "multi-paragraph subtopic passages. Computational Linguistics.\n"
    "[2] Lewis, P. et al. (2020). Retrieval-Augmented Generation for "
    "Knowledge-Intensive NLP Tasks. NeurIPS.\n"
    "[3] Karpukhin, V. et al. (2020). Dense Passage Retrieval for "
    "Open-Domain Question Answering. EMNLP."
)

PAGE_FOOTER = "1"


def wrap_text(text, width_chars=95):
    """Very simple word-wrap so reportlab doesn't overflow the page width."""
    lines = []
    for paragraph in text.split("\n\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= width_chars:
                current = f"{current} {word}".strip()
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        lines.append("")  # blank line between paragraphs
    return lines


def draw_page_header_footer(c, page_num):
    c.setFont("Helvetica", 7)
    c.drawString(0.75 * inch, 10.6 * inch, PAGE_HEADER)
    c.drawCentredString(4.25 * inch, 0.5 * inch, str(page_num))


def build_pdf():
    c = canvas.Canvas(OUTPUT_PATH, pagesize=letter)
    page_num = 1
    y = 10.2 * inch
    left_margin = 0.85 * inch
    line_height = 13

    draw_page_header_footer(c, page_num)

    # Title
    c.setFont("Helvetica-Bold", 15)
    title_lines = wrap_text(TITLE, width_chars=55)
    for line in title_lines:
        c.drawCentredString(4.25 * inch, y, line)
        y -= 18

    y -= 6
    c.setFont("Helvetica", 10)
    c.drawCentredString(4.25 * inch, y, AUTHORS)
    y -= 14
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(4.25 * inch, y, AFFIL)
    y -= 26

    # Abstract
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_margin, y, "Abstract")
    y -= line_height
    c.setFont("Helvetica", 9.5)
    for line in wrap_text(ABSTRACT):
        if y < 0.9 * inch:
            c.showPage()
            page_num += 1
            draw_page_header_footer(c, page_num)
            y = 10.2 * inch
        c.drawString(left_margin, y, line)
        y -= line_height

    y -= 10

    # Sections
    for heading, body in SECTIONS:
        if y < 1.3 * inch:
            c.showPage()
            page_num += 1
            draw_page_header_footer(c, page_num)
            y = 10.2 * inch

        c.setFont("Helvetica-Bold", 11)
        c.drawString(left_margin, y, heading)
        y -= line_height + 2

        c.setFont("Helvetica", 9.5)
        for line in wrap_text(body):
            if y < 0.9 * inch:
                c.showPage()
                page_num += 1
                draw_page_header_footer(c, page_num)
                y = 10.2 * inch
            c.drawString(left_margin, y, line)
            y -= line_height
        y -= 8

    # References
    if y < 2 * inch:
        c.showPage()
        page_num += 1
        draw_page_header_footer(c, page_num)
        y = 10.2 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_margin, y, "References")
    y -= line_height + 2
    c.setFont("Helvetica", 9)
    for line in REFERENCES.split("\n")[1:]:
        for wrapped in wrap_text(line, width_chars=95):
            if y < 0.9 * inch:
                c.showPage()
                page_num += 1
                draw_page_header_footer(c, page_num)
                y = 10.2 * inch
            c.drawString(left_margin, y, wrapped)
            y -= line_height

    c.save()
    print(f"Wrote test PDF to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
