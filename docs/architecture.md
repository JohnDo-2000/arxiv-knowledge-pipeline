# Architecture

## Overview

```
arXiv API
    |  (search by category/query, download PDFs)
    v
fetch_arxiv.py
    |  (upload PDF + metadata.json sidecar)
    v
Amazon S3  (raw/run_date=YYYY-MM-DD/<arxiv_id>.pdf)
    |
    |  S3 ObjectCreated:Put event (filtered to *.pdf)
    v
AWS Lambda (container image)
    |
    |-- 1. parse.py     : unstructured.io partitions the PDF into typed
    |                      elements, strips headers/footers/page numbers
    |
    |-- 2. chunk.py      : LangChain RecursiveCharacterTextSplitter splits
    |                      cleaned text into ~1000-char chunks on natural
    |                      boundaries (paragraph > line > sentence > word)
    |
    |-- 3. embed.py      : sentence-transformers/all-MiniLM-L6-v2 generates
    |                      384-dim normalized embeddings, locally, free
    |
    v
qdrant_client.py
    |  (upsert vector + metadata payload per chunk)
    v
Qdrant  (collection: arxiv_papers)
    |
    v
query_demo.py  (semantic search + metadata filters: category, author, date range)
```

## Why these choices

### Event-driven (S3 trigger), not scheduled (cron)

The companion job-market-analysis project uses EventBridge on a daily
schedule, which is the right fit for "pull a fresh snapshot of an API
once a day." This project's natural trigger is different: documents
arrive at unpredictable times (whenever fetch_arxiv.py runs, or
whenever someone manually drops a file in the bucket), so processing
should react to that arrival rather than wait for the next cron tick.
This is also a deliberately different AWS pattern to show range across
the two projects.

### Lambda container image, not zip deployment

`unstructured`, `sentence-transformers`, and `torch` together exceed
Lambda's 250MB zip size limit by a wide margin, and some of their
dependencies are native (C extensions) that need to match the Lambda
execution environment exactly. Container images solve both problems
and support up to 10GB.

### Models baked into the image at build time

Both `unstructured` (via a spaCy model) and the embedding step (via a
HuggingFace sentence-transformers model) need to download files on
first use. Doing that download inside a running Lambda function would
mean:
  - Every cold start depends on external network access succeeding
  - Cold start latency balloons (downloading ~100-500MB models)
  - A transient network blip becomes a pipeline failure

Baking both into the Docker image at build time (see
`infra/Dockerfile.lambda`) makes cold starts fully self-contained and
deterministic.

### Qdrant self-hosted, not a managed SaaS vector DB

Pinecone-as-a-service would work, but the entire interaction would be
"call an API with a key" -- there's no infrastructure story to tell.
Running Qdrant via Docker (locally for dev, on a small EC2 instance or
your own server for "production") means there's real infra you stood
up and can speak to in an interview: container networking, persistent
volume storage, exposing the right ports, etc.

### HuggingFace sentence-transformers, not OpenAI/Cohere embeddings

Free, runs locally with no API key or per-call cost, and demonstrates
understanding of running an actual ML model rather than just calling a
hosted endpoint. `all-MiniLM-L6-v2` specifically is small enough
(~80MB) to comfortably bake into a Lambda container image.

### Metadata-first chunk payloads

Every chunk's Qdrant payload carries the full document metadata
(arxiv_id, title, authors, category, published_date, plus a derived
published_epoch for numeric range filtering) in addition to its own
chunk_index and char_count. This is what makes filtered retrieval
queries possible ("only cs.CL papers from after January 2024") without
needing a separate metadata database or join.
