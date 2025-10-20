"""
app/main.py: RAG Backend Service entry point

Endpoints:
  POST   /documents                : ingest a document
  GET    /documents                : list documents in a collection
  GET    /documents/{id}           : get document details
  DELETE /documents/{id}           : delete document + its chunks

  POST   /search                   : hybrid or vector search with reranking

  POST   /collections              : create a collection (namespace)
  GET    /collections              : list all collections
  DELETE /collections/{name}       : delete a collection

  GET    /health                   : liveness check
  GET    /stats                    : ingestion and search stats
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import documents, search, collections
from app.database import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("RAG backend service starting...")
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        log.info("DB connection OK")
    except Exception as e:
        log.error("DB connection failed: %s", e)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="RAG Backend Service",
    description=(
        "Production-grade retrieval-augmented generation backend. "
        "Hybrid vector + BM25 search with multi-stage LLM reranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(collections.router)


@app.get("/health")
def health():
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "unreachable"
    return {"status": "ok" if db_status == "ok" else "degraded", "db": db_status}


@app.get("/stats")
def stats():
    from app.database import query as db_query
    doc_stats = db_query("""
        SELECT
            COUNT(*)                                            AS total_documents,
            SUM(chunk_count)                                    AS total_chunks,
            COUNT(DISTINCT collection_id)                       AS collections_used,
            SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END)  AS ready,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)  AS errors
        FROM documents
    """)[0]

    search_stats = db_query("""
        SELECT
            COUNT(*)                        AS total_searches,
            ROUND(AVG(latency_ms))          AS avg_latency_ms,
            ROUND(AVG(result_count), 1)     AS avg_results
        FROM search_log
    """)[0]

    return {"documents": doc_stats, "search": search_stats}
