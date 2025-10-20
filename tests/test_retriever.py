"""
tests/test_retriever.py

Unit tests for Reciprocal Rank Fusion (RRF).

RRF is the core fusion algorithm that merges vector and BM25 ranked lists.
It is pure Python with no I/O — these tests run without a DB or OpenAI key.

Reference: Cormack et al. 2009 — k=60 is the standard constant.
"""

import pytest
from app.services.retriever import reciprocal_rank_fusion


# -- Fixtures ----------------------------------------------------------------

def _vec(chunk_id: str, score: float) -> dict:
    """Minimal vector search result."""
    return {"chunk_id": chunk_id, "vector_score": score, "content": f"chunk {chunk_id}"}


def _bm25(chunk_id: str, score: float) -> dict:
    """Minimal BM25 search result."""
    return {"chunk_id": chunk_id, "bm25_score": score, "content": f"chunk {chunk_id}"}


# -- Tests -------------------------------------------------------------------

class TestReciprocalRankFusion:

    def test_output_sorted_by_hybrid_score_descending(self):
        vec  = [_vec("a", 0.9), _vec("b", 0.7), _vec("c", 0.5)]
        bm25 = [_bm25("a", 3.0), _bm25("d", 2.0)]
        merged = reciprocal_rank_fusion(vec, bm25, alpha=0.7)
        scores = [r["hybrid_score"] for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_in_both_lists_scores_higher_than_exclusive(self):
        """A chunk present in both lists must outscore chunks from only one."""
        vec  = [_vec("shared", 0.8), _vec("vec_only", 0.7)]
        bm25 = [_bm25("shared", 2.0), _bm25("bm25_only", 1.5)]
        merged = reciprocal_rank_fusion(vec, bm25, alpha=0.5)
        by_id = {r["chunk_id"]: r["hybrid_score"] for r in merged}
        assert by_id["shared"] > by_id.get("vec_only", 0)
        assert by_id["shared"] > by_id.get("bm25_only", 0)

    def test_alpha_1_weights_vector_rank_only(self):
        """alpha=1.0 — BM25 rank contributes nothing to the hybrid score."""
        vec  = [_vec("a", 0.9), _vec("b", 0.5)]
        bm25 = [_bm25("b", 10.0)]  # b dominates BM25 but alpha=1 ignores it
        merged = reciprocal_rank_fusion(vec, bm25, alpha=1.0)
        by_id = {r["chunk_id"]: r["hybrid_score"] for r in merged}
        # a ranked 1st in vector — with alpha=1 it must outscore b
        assert by_id["a"] > by_id["b"]

    def test_alpha_0_weights_bm25_rank_only(self):
        """alpha=0.0 — vector rank contributes nothing to the hybrid score."""
        vec  = [_vec("a", 0.9)]               # a dominates vector
        bm25 = [_bm25("b", 5.0), _bm25("a", 1.0)]  # b ranked 1st in BM25
        merged = reciprocal_rank_fusion(vec, bm25, alpha=0.0)
        by_id = {r["chunk_id"]: r["hybrid_score"] for r in merged}
        assert by_id["b"] > by_id["a"]

    def test_empty_bm25_list_handled(self):
        vec = [_vec("a", 0.9), _vec("b", 0.7)]
        merged = reciprocal_rank_fusion(vec, [], alpha=0.7)
        assert len(merged) == 2
        assert all(r["hybrid_score"] > 0 for r in merged)

    def test_empty_vector_list_handled(self):
        bm25 = [_bm25("x", 3.0), _bm25("y", 1.5)]
        merged = reciprocal_rank_fusion([], bm25, alpha=0.7)
        assert len(merged) == 2

    def test_both_empty_returns_empty(self):
        assert reciprocal_rank_fusion([], [], alpha=0.7) == []

    def test_output_contains_union_of_all_chunk_ids(self):
        vec  = [_vec("a", 0.9), _vec("b", 0.7)]
        bm25 = [_bm25("b", 2.0), _bm25("c", 1.5)]
        merged = reciprocal_rank_fusion(vec, bm25, alpha=0.7)
        assert {r["chunk_id"] for r in merged} == {"a", "b", "c"}

    def test_rrf_score_uses_k60_constant(self):
        """Rank-1 RRF score must equal 1/(60+1) per the Cormack et al. formula."""
        vec = [_vec("only", 1.0)]
        merged = reciprocal_rank_fusion(vec, [], alpha=1.0)
        expected = 1.0 / (60 + 1)
        assert abs(merged[0]["hybrid_score"] - expected) < 1e-9

    def test_higher_ranked_item_gets_higher_rrf_contribution(self):
        """Rank 1 must have a higher RRF contribution than rank 2."""
        vec = [_vec("first", 0.9), _vec("second", 0.8)]
        merged = reciprocal_rank_fusion(vec, [], alpha=1.0)
        by_id = {r["chunk_id"]: r["hybrid_score"] for r in merged}
        assert by_id["first"] > by_id["second"]
