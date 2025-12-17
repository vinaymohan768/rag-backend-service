"""
routers/search.py — search and retrieval endpoints
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from openai import OpenAI

from app.config import settings
from app.database import query as db_query, execute
from app.models.schemas import SearchRequest, SearchResponse
from app.services.retriever import search as run_search

log = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


def get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _resolve_collection(name: str) -> str:
    rows = db_query("SELECT id FROM collections WHERE name = %s", (name,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return str(rows[0]["id"])


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest, client: OpenAI = Depends(get_client)):
    collection_id = _resolve_collection(req.collection)

    results, latency_ms = run_search(
        query=req.query,
        collection_id=collection_id,
        client=client,
        top_k=req.top_k,
        rerank_k=req.rerank_k,
        strategy=req.strategy.value,
        source_filter=req.source_filter,
    )

    # Log the search for analytics
    try:
        execute(
            """
            INSERT INTO search_log
                (collection_id, query, top_k, rerank_k, strategy, result_count, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                collection_id, req.query,
                req.top_k or settings.default_top_k,
                req.rerank_k or settings.default_rerank_top_k,
                req.strategy.value, len(results), latency_ms,
            ),
        )
    except Exception:
        pass  # Don't fail search requests due to logging errors

    return SearchResponse(
        query=req.query,
        collection=req.collection,
        strategy=req.strategy.value,
        results=results,
        result_count=len(results),
        latency_ms=latency_ms,
    )
