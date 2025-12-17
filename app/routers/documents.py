"""
routers/documents.py — document ingestion and management
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends
from openai import OpenAI

from app.config import settings
from app.database import query as db_query, execute, execute_values
from app.models.schemas import IngestRequest, IngestResponse, DocumentResponse
from app.services.chunker import chunk_text, token_count, tokenize_for_bm25
from app.services.embeddings import embed_texts

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _resolve_collection(name: str) -> str:
    rows = db_query("SELECT id FROM collections WHERE name = %s", (name,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return str(rows[0]["id"])


@router.post("", response_model=IngestResponse, status_code=201)
def ingest_document(req: IngestRequest, client: OpenAI = Depends(get_client)):
    collection_id = _resolve_collection(req.collection)

    chunk_size = req.chunk_size or settings.default_chunk_size
    chunk_overlap = req.chunk_overlap or settings.default_chunk_overlap

    chunks = chunk_text(
        req.text,
        strategy=req.chunk_strategy.value,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No content could be extracted from the provided text.")

    # Create document record
    doc_id = str(uuid.uuid4())
    execute(
        """
        INSERT INTO documents
            (id, collection_id, title, source, char_count, chunk_count, chunk_strategy, status, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'processing', %s::jsonb)
        """,
        (doc_id, collection_id, req.title, req.source, len(req.text),
         len(chunks), req.chunk_strategy.value, str(req.metadata).replace("'", '"')),
    )

    try:
        embeddings = embed_texts(chunks, client)

        rows = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            tokens = tokenize_for_bm25(chunk)
            rows.append((
                str(uuid.uuid4()), doc_id, collection_id, i,
                chunk, token_count(chunk), embedding, tokens,
            ))

        execute_values(
            """
            INSERT INTO chunks
                (id, document_id, collection_id, chunk_index, content, token_count, embedding, tokens)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s::vector, %s)",
        )

        execute(
            "UPDATE documents SET status = 'ready', chunk_count = %s WHERE id = %s",
            (len(chunks), doc_id),
        )
        log.info("Ingested doc_id=%s chunks=%d collection=%s", doc_id, len(chunks), req.collection)

    except Exception as e:
        execute(
            "UPDATE documents SET status = 'error', error_message = %s WHERE id = %s",
            (str(e), doc_id),
        )
        log.error("Ingestion failed for doc_id=%s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return IngestResponse(
        document_id=doc_id,
        collection=req.collection,
        title=req.title,
        chunk_count=len(chunks),
        char_count=len(req.text),
        chunk_strategy=req.chunk_strategy.value,
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(collection: str = "default", limit: int = 50):
    collection_id = _resolve_collection(collection)
    rows = db_query(
        """
        SELECT id, collection_id, title, source, char_count, chunk_count,
               chunk_strategy, status, metadata, created_at
        FROM documents
        WHERE collection_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (collection_id, limit),
    )
    return [DocumentResponse(**{**r, "id": str(r["id"]), "collection_id": str(r["collection_id"])}) for r in rows]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str):
    rows = db_query(
        """
        SELECT id, collection_id, title, source, char_count, chunk_count,
               chunk_strategy, status, metadata, created_at
        FROM documents WHERE id = %s
        """,
        (document_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    r = rows[0]
    return DocumentResponse(**{**r, "id": str(r["id"]), "collection_id": str(r["collection_id"])})


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str):
    rows = db_query("SELECT id FROM documents WHERE id = %s", (document_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")
    execute("DELETE FROM documents WHERE id = %s", (document_id,))
