"""
services/retriever.py

Two-stage retrieval pipeline:

  Stage 1: Candidate retrieval
    Vector search: cosine similarity via pgvector IVFFlat index
    BM25 search:   keyword overlap using stored token arrays (hybrid mode)
    Fusion:        Reciprocal Rank Fusion (RRF) to combine vector + BM25 scores

  Stage 2: Reranking
    LLM scores each candidate 0-10 for relevance, blends with RRF score.
    Falls back to hybrid order if the LLM call fails.
"""

import json
import logging
import time
from openai import OpenAI

from app.config import settings
from app.database import query as db_query, execute
from app.services.embeddings import embed_query
from app.models.schemas import ChunkResult

log = logging.getLogger(__name__)


# ── Vector search ─────────────────────────────────────────────────────────────

def vector_search(
    query_embedding: list[float],
    collection_id: str,
    top_k: int,
    source_filter: str | None = None,
) -> list[dict]:
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

    sql = """
        SELECT
            c.id            AS chunk_id,
            c.document_id,
            c.chunk_index,
            c.content,
            c.tokens,
            d.title         AS document_title,
            d.source,
            1 - (c.embedding <=> %s::vector) AS vector_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.collection_id = %s
          AND c.embedding IS NOT NULL
    """
    params: list = [embedding_str, collection_id]

    if source_filter:
        sql += " AND d.source = %s"
        params.append(source_filter)

    sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
    params.extend([embedding_str, top_k])

    return db_query(sql, params)


# ── BM25 search ───────────────────────────────────────────────────────────────

def bm25_search(
    query_tokens: list[str],
    collection_id: str,
    top_k: int,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Approximate BM25 using PostgreSQL GIN index on the tokens array.
    Scores by overlap count (term frequency proxy): fast and DB-native.
    For production, consider pg_search or a dedicated BM25 extension.
    """
    if not query_tokens:
        return []

    sql = """
        SELECT
            c.id            AS chunk_id,
            c.document_id,
            c.chunk_index,
            c.content,
            d.title         AS document_title,
            d.source,
            (
                SELECT COUNT(*)
                FROM unnest(c.tokens) t
                WHERE t = ANY(%s)
            )::float AS bm25_score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.collection_id = %s
          AND c.tokens && %s
    """
    params: list = [query_tokens, collection_id, query_tokens]

    if source_filter:
        sql += " AND d.source = %s"
        params.append(source_filter)

    sql += " ORDER BY bm25_score DESC LIMIT %s"
    params.append(top_k)

    return db_query(sql, params)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    alpha: float,
    k: int = 60,
) -> list[dict]:
    """
    RRF merges two ranked lists into one without requiring score normalization.
    RRF(d) = alpha * (1 / (k + rank_vector)) + (1-alpha) * (1 / (k + rank_bm25))

    k=60 is the standard constant from the original RRF paper (Cormack et al., 2009).
    """
    scores: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, start=1):
        cid = str(r["chunk_id"])
        scores[cid] = {**r, "vector_score": r["vector_score"], "bm25_score": 0.0}
        scores[cid]["rrf_vector"] = 1.0 / (k + rank)

    for rank, r in enumerate(bm25_results, start=1):
        cid = str(r["chunk_id"])
        if cid not in scores:
            scores[cid] = {**r, "vector_score": 0.0, "bm25_score": r.get("bm25_score", 0.0)}
            scores[cid]["rrf_vector"] = 0.0
        else:
            scores[cid]["bm25_score"] = r.get("bm25_score", 0.0)
        scores[cid]["rrf_bm25"] = 1.0 / (k + rank)

    for cid, data in scores.items():
        data["hybrid_score"] = (
            alpha * data.get("rrf_vector", 0.0)
            + (1 - alpha) * data.get("rrf_bm25", 0.0)
        )

    merged = sorted(scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return merged


# ── LLM reranking ─────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int, client: OpenAI) -> list[dict]:
    """
    Ask the LLM to score each candidate's relevance to the query (0-10).
    Falls back to hybrid score ordering if the LLM call fails or returns
    malformed output.
    """
    if not candidates:
        return []

    passages = "\n\n".join(
        f"[{i+1}] {c['content'][:400]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Query: {query}\n\n"
        "Rate each passage's relevance to the query from 0 (irrelevant) to 10 (highly relevant).\n"
        "Consider factual relevance, specificity, and information density.\n"
        "Respond with ONLY a JSON array of numbers, one per passage.\n\n"
        f"{passages}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        scores = json.loads(raw)
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError("Score list length mismatch")
    except Exception as e:
        log.warning("Rerank failed (%s), using hybrid order", e)
        for c in candidates[:top_k]:
            c["rerank_score"] = None
            c["final_score"] = c.get("hybrid_score") or c.get("vector_score", 0.0)
        return candidates[:top_k]

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
        # Blend rerank score with hybrid score for final ranking
        candidate["final_score"] = 0.6 * float(score) / 10.0 + 0.4 * candidate.get("hybrid_score", candidate.get("vector_score", 0.0))

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]


# ── Main search pipeline ──────────────────────────────────────────────────────

def search(
    query: str,
    collection_id: str,
    client: OpenAI,
    top_k: int = None,
    rerank_k: int = None,
    strategy: str = "hybrid",
    source_filter: str | None = None,
) -> tuple[list[ChunkResult], int]:
    top_k = top_k or settings.default_top_k
    rerank_k = rerank_k or settings.default_rerank_top_k
    alpha = settings.hybrid_alpha

    t0 = time.monotonic()

    query_embedding = embed_query(query, client)

    if strategy == "hybrid":
        from app.services.chunker import tokenize_for_bm25
        query_tokens = tokenize_for_bm25(query)
        vec_results = vector_search(query_embedding, collection_id, top_k, source_filter)
        bm25_results = bm25_search(query_tokens, collection_id, top_k, source_filter)
        candidates = reciprocal_rank_fusion(vec_results, bm25_results, alpha)
    else:
        candidates = vector_search(query_embedding, collection_id, top_k, source_filter)
        for c in candidates:
            c["hybrid_score"] = c["vector_score"]
            c["bm25_score"] = None

    ranked = rerank(query, candidates[:top_k], rerank_k, client)

    latency_ms = int((time.monotonic() - t0) * 1000)
    log.info("Search complete | strategy=%s candidates=%d reranked=%d latency=%dms",
             strategy, len(candidates), len(ranked), latency_ms)

    results = [
        ChunkResult(
            chunk_id=str(r["chunk_id"]),
            document_id=str(r["document_id"]),
            document_title=r.get("document_title", ""),
            source=r.get("source", ""),
            content=r["content"],
            chunk_index=r.get("chunk_index", 0),
            vector_score=round(float(r.get("vector_score") or 0), 4),
            bm25_score=round(float(r["bm25_score"]), 4) if r.get("bm25_score") is not None else None,
            hybrid_score=round(float(r.get("hybrid_score") or 0), 4),
            rerank_score=round(float(r["rerank_score"]), 2) if r.get("rerank_score") is not None else None,
            final_score=round(float(r.get("final_score") or 0), 4),
        )
        for r in ranked
    ]

    return results, latency_ms
