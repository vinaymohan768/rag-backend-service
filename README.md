# rag-backend-service

![CI](https://github.com/vinaymohan768/rag-backend-service/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-0.3-4169E1)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

Production-grade RAG backend service with hybrid vector + BM25 search, multi-stage LLM reranking, and collection-based document namespacing. Built with FastAPI, pgvector, and OpenAI.

**What makes this different from basic RAG:** hybrid search catches exact keyword matches that pure vector search misses, and LLM reranking re-scores top candidates for relevance before returning results — standard approach for reducing hallucinated outputs in enterprise knowledge retrieval.

---

## Architecture

```
                        Client
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │  :8000
                    │             │
                    │ /documents  │  — ingest, list, delete
                    │ /search     │  — query with hybrid retrieval
                    │ /collections│  — namespace management
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────────┐
   │   Chunker   │  │  Embeddings │  │   Retriever    │
   │             │  │             │  │                │
   │ fixed       │  │ OpenAI      │  │ Vector search  │
   │ sentence ✓  │  │ text-emb-   │  │ BM25 search    │
   │ paragraph   │  │ 3-small     │  │ RRF fusion     │
   └──────┬──────┘  └──────┬──────┘  │ LLM reranking  │
          │                │         └────────┬───────┘
          └────────────────▼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ PostgreSQL  │
                    │ + pgvector  │
                    │             │
                    │ collections │
                    │ documents   │
                    │ chunks      │  ← IVFFlat index + GIN token index
                    │ search_log  │
                    └─────────────┘
```

---

## Search Pipeline

```
Query
  │
  ├─── OpenAI embedding (text-embedding-3-small)
  │         │
  │    Vector search (cosine ANN via IVFFlat)
  │         │
  │         ├── top-K vector candidates
  │
  ├─── BM25 tokenization
  │         │
  │    Keyword search (GIN index on token arrays)
  │         │
  │         ├── top-K BM25 candidates
  │
  ▼
Reciprocal Rank Fusion (alpha=0.7)
  │   merges both lists without score normalization
  │
  ▼
LLM Reranking (gpt-4o-mini)
  │   scores each candidate 0-10 for relevance
  │   blends with RRF score: 0.6 * rerank + 0.4 * hybrid
  │
  ▼
Final top-K results with full score breakdown
```

**Why hybrid over pure vector?**
Vector search excels at semantic similarity but misses exact matches — product codes, proper nouns, technical identifiers. BM25 catches these but misses paraphrases. Hybrid at alpha=0.7 consistently outperforms either alone on retrieval benchmarks (BEIR, MTEB).

---

## Chunking Strategies

| Strategy | How it splits | Best for |
|---|---|---|
| `sentence` (default) | Sentence boundaries, packs to token limit with overlap | General prose, documentation |
| `paragraph` | Double-newlines first, sentence-splits oversized paragraphs | Long-form content with clear sections |
| `fixed` | Naive token-count slicing | Structured data, code, tables |

All strategies use tiktoken `cl100k_base` for accurate token counting.

---

## Getting Started

```bash
git clone https://github.com/vinaymohan768/rag-backend-service
cd rag-backend-service

cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker compose up --build
```

API at `http://localhost:8000` · Swagger UI at `http://localhost:8000/docs`

---

## Usage Examples

**Create a collection:**
```bash
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "product-docs", "description": "Product documentation"}'
```

**Ingest a document:**
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "API Reference Guide",
    "text": "Our API supports REST and GraphQL endpoints...",
    "source": "api-reference-v2",
    "collection": "product-docs",
    "chunk_strategy": "sentence"
  }'
```

**Hybrid search with reranking:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I authenticate API requests?",
    "collection": "product-docs",
    "strategy": "hybrid",
    "top_k": 8,
    "rerank_k": 3
  }'
```

**Response includes full score breakdown:**
```json
{
  "results": [
    {
      "content": "Authentication uses Bearer tokens...",
      "vector_score": 0.8821,
      "bm25_score": 0.6,
      "hybrid_score": 0.0156,
      "rerank_score": 9.0,
      "final_score": 0.5462
    }
  ],
  "latency_ms": 312
}
```

---

## Project Structure

```
rag-backend-service/
├── app/
│   ├── main.py               # FastAPI app, /health, /stats
│   ├── config.py             # Settings from env vars
│   ├── database.py           # psycopg2 connection + helpers
│   ├── routers/
│   │   ├── documents.py      # Ingest + document CRUD
│   │   ├── search.py         # Hybrid search endpoint
│   │   └── collections.py    # Namespace management
│   ├── services/
│   │   ├── chunker.py        # Three chunking strategies
│   │   ├── embeddings.py     # OpenAI batch embeddings
│   │   └── retriever.py      # Vector search + BM25 + RRF + reranking
│   └── models/
│       └── schemas.py        # Pydantic request/response models
├── db/
│   └── init.sql              # Schema: IVFFlat + GIN indexes, search_log
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Tech Stack

`Python 3.11` `FastAPI` `pgvector` `PostgreSQL 16` `OpenAI API` `BM25` `Docker Compose`
