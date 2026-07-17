# arXiv Knowledge Pipeline

### Event-Driven ETL for LLM Retrieval (Unstructured Data → Vector DB)

An automated pipeline that ingests unstructured PDF documents (arXiv
research papers), parses and cleans them, breaks them into semantic
chunks, generates vector embeddings using a free open-source model, and
upserts them into a vector database with rich metadata for filtered
retrieval — the kind of ingestion pipeline that sits behind any
production RAG system.

## Architecture

```
arXiv API (e.g. cs.LG, cs.CL, cs.CV categories)
        |
        v
fetch_arxiv.py  -->  Amazon S3 (raw landing zone, partitioned by run_date)
                              |
                              | S3 ObjectCreated event (*.pdf suffix filter)
                              v
                     AWS Lambda (container image)
                              |
                   unstructured.io --> parse + clean PDF
                              |
                   LangChain --> semantic chunking (~1000 chars, 150 overlap)
                              |
                   sentence-transformers --> 384-dim embeddings (free, local)
                              |
                              v
                     Qdrant (self-hosted, Docker)
                              |
                   Payload filters: category, author, published date
                              v
                     Semantic search + filtered retrieval demo
```

Full design rationale in [`docs/architecture.md`](docs/architecture.md).

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, arXiv API |
| Storage (raw landing zone) | Amazon S3 (partitioned by `run_date=YYYY-MM-DD`) |
| Trigger | S3 ObjectCreated event notification (event-driven, not scheduled) |
| Compute | AWS Lambda (container image — required for ML dep size) |
| Parsing / cleaning | unstructured.io (header/footer/boilerplate removal) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` — free, local, no API key |
| Vector store | Qdrant (self-hosted via Docker) |
| Metadata filtering | Qdrant payload filters (category, author, published date range) |

## Why this pairs with my Texas Job Market Pipeline

This project deliberately covers different ground:

| | Texas Job Market Pipeline | This Project |
|---|---|---|
| Data shape | Structured JSON/CSV | Unstructured PDFs |
| Trigger | Scheduled (EventBridge cron, daily 8am UTC) | Event-driven (S3 ObjectCreated) |
| Core logic | SQL (Athena UNION ALL queries) | Python (parsing, chunking, ML embeddings) |
| Output | Tableau BI dashboard | Vector store for semantic retrieval |
| New concepts | Data cataloging (Glue), BI | Embeddings, vector search, semantic chunking |

## Pipeline Results (Validated End-to-End)

From a real run against 5 arXiv cs.LG papers (July 2026):

- **5 papers ingested** from arXiv API → uploaded to S3 with metadata sidecars
- **520 chunks upserted** into Qdrant (~104 chunks/paper average)
- **Chunk size**: ~1000 characters, 150-character overlap
- **Embedding model**: `all-MiniLM-L6-v2`, 384 dimensions, normalized
- **Parsing**: 17 raw PDF elements → 13 kept text blocks per paper (boilerplate filtered)

### Sample Semantic Search

Query: `"how do language models handle retrieval augmentation"`
```
1. score=0.4914  [cs.CL]  Mask-Aware Policy Gradients for Diffusion Language Models
2. score=0.4574  [cs.RO]  RoboTTT: Context Scaling for Robot Policies
3. score=0.4545  [cs.CV]  MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators
```

Query: `"policy gradient optimization"` filtered to `--category cs.LG`
```
1. score=0.4397  [cs.LG]  BadWAM: When World-Action Models Dream Right but Act Wrong
2. score=0.4368  [cs.LG]  BadWAM: When World-Action Models Dream Right but Act Wrong
3. score=0.4118  [cs.LG]  BadWAM: When World-Action Models Dream Right but Act Wrong
```
*Category filter correctly excluded cs.CL, cs.CV, cs.RO papers from results.*

## S3 Structure

```
s3://arxiv-knowledge-pipeline-dotriet-2026/
└── raw/
    └── run_date=2026-07-17/
        ├── 2607.15200.pdf
        ├── 2607.15200.metadata.json
        ├── 2607.15207.pdf
        ├── 2607.15207.metadata.json
        └── ...
```

## Key Engineering Decisions

**Event-driven trigger (not cron):** The companion job-market pipeline
uses a daily EventBridge schedule. This pipeline fires the moment a PDF
lands in S3 — demonstrating a distinct, real-world trigger pattern.

**Lambda container image (not zip):** `unstructured` + `sentence-transformers`
+ `torch` exceed Lambda's 250MB zip limit. Container images support up
to 10GB. Both the spaCy NLP model (used by unstructured for element
classification) and the embedding model are baked into the image at
build time — fetching them at runtime would mean network dependency
on every cold start.

**Qdrant self-hosted (not Pinecone SaaS):** Running Qdrant via Docker
means there's real infrastructure to speak to in an interview: container
networking, persistent volume mounts, port exposure. Pinecone is one
API key away from a managed service — Qdrant is a system you stood up.

**Metadata-first chunk payloads:** Every chunk carries full document
metadata (arxiv_id, title, authors, category, published_date, and a
derived Unix-epoch timestamp for numeric range filtering). This enables
filtered retrieval without a separate metadata database or join.

## Getting Started (local, no AWS required)

```bash
git clone <this-repo>
cd arxiv-knowledge-pipeline

pip install -r requirements.txt
python -m spacy download en_core_web_sm

docker compose up -d   # starts Qdrant on localhost:6333

# Add PDFs to data/sample_pdfs/, then:
python scripts/run_local_demo.py

# Query the collection:
python scripts/query_demo.py "your query here"
python scripts/query_demo.py "your query" --category cs.LG --after 2024-01-01
```

## AWS Deployment

See [`infra/deploy_notes.md`](infra/deploy_notes.md) for full
step-by-step: S3 bucket, Qdrant hosting options, Lambda container image
build/push, and S3 event trigger setup.

## Repo Structure

```
arxiv-knowledge-pipeline/
├── src/
│   ├── ingest/fetch_arxiv.py      # arXiv API → S3 raw landing zone
│   ├── etl/
│   │   ├── parse.py               # unstructured.io parsing + cleaning
│   │   ├── chunk.py               # LangChain semantic chunking
│   │   ├── embed.py               # sentence-transformers embeddings
│   │   └── pipeline.py            # orchestrates parse→chunk→embed→upsert
│   ├── vectorstore/qdrant_client.py
│   └── lambda_handler.py          # S3-event-triggered Lambda entrypoint
├── scripts/
│   ├── run_local_demo.py          # end-to-end local run (no AWS needed)
│   └── query_demo.py              # semantic search + metadata filter demo
├── infra/
│   ├── Dockerfile.lambda          # container image for Lambda deployment
│   └── deploy_notes.md
├── docs/architecture.md
└── docker-compose.yml             # local Qdrant
```

---

## Resume Bullets

```
arXiv Knowledge Pipeline | Python, unstructured.io, LangChain,
sentence-transformers, Qdrant, AWS S3, Lambda

• Built an event-driven ETL pipeline that ingests unstructured arXiv
  PDFs from S3, parses and cleans them with unstructured.io (filtering
  headers/footers/boilerplate), chunks with LangChain's
  RecursiveCharacterTextSplitter, and generates 384-dim embeddings
  with HuggingFace sentence-transformers — no paid API required

• Upserted 520 vector chunks from 5 real arXiv papers (~104 chunks/paper)
  into a self-hosted Qdrant vector store with rich metadata payloads
  (arxiv_id, authors, category, published_date), enabling filtered
  semantic search by category, author, and date range

• Deployed pipeline as an AWS Lambda container image (vs. zip) to
  accommodate ML dependencies exceeding the 250MB zip limit; baked
  spaCy and sentence-transformer models into the image at build time
  to eliminate cold-start network dependency on HuggingFace/GitHub

• Implemented S3 ObjectCreated event trigger (vs. scheduled cron in
  companion job-market pipeline), demonstrating event-driven pipeline
  architecture alongside traditional batch ELT patterns
```
