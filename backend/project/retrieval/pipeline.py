"""
End-to-end retrieval pipeline:
"""

from __future__ import annotations

from typing import List, Tuple
from flask import current_app
import config as cfg
from project.vector_db.redis_db import RedisDB
from project.retrieval.rrf import reciprocal_rank_fusion
from project.retrieval.reranker import rerank as cross_encoder_rerank
from project.observability import current_trace


def _summarize_hits(docs, score_attr: str = "vector_score", limit: int = 5):
    """Compact (file_name, score) summary for tracing — first `limit` only."""
    out = []
    for d in docs[:limit]:
        try:
            file_name = getattr(d, "file_name", None) or d.__dict__.get("file_name", "")
            score = getattr(d, score_attr, None) or d.__dict__.get(score_attr)
            out.append({"id": d.id, "file_name": file_name, "score": float(score) if score is not None else None})
        except Exception:
            out.append({"id": getattr(d, "id", "?")})
    return out


def _attach_neighbors(redis_obj: RedisDB, docs: List[dict]) -> List[dict]:
    """Optional: pull prev_id / next_id chunks and inline them as extra context docs."""
    if not docs:
        return docs
    extra_ids = []
    seen = {d["id"] for d in docs}
    for d in docs:
        for k in ("prev_id", "next_id"):
            nid = d.get(k)
            if nid and nid not in seen:
                extra_ids.append(nid)
                seen.add(nid)
    if extra_ids:
        neighbors = redis_obj.get_docs_by_ids(extra_ids)
        docs.extend(neighbors)
    return docs


def retrieve(question: str, index_name: str) -> List[dict]:
    """
    Run the full retrieval pipeline and return final docs (list of dicts with
    `file_name`, `content`, `heading`, etc.) ready for prompt injection.

    Trace fields populated: vector_hits, bm25_hits, rrf_top, rerank_top, final_doc_ids
    and per-stage latency.
    """
    trace = current_trace()
    redis_obj = RedisDB()

    # ------------------------------------------------------------------
    # Stage 1: candidate retrieval
    # ------------------------------------------------------------------
    vector_docs = []
    bm25_docs = []

    if trace:
        with trace.timer("vector"):
            vector_docs = redis_obj.vector_search(question, index_name)
    else:
        vector_docs = redis_obj.vector_search(question, index_name)

    if cfg.HYBRID_SEARCH_ENABLED:
        if trace:
            with trace.timer("bm25"):
                bm25_docs = redis_obj.bm25_search(question, index_name)
        else:
            bm25_docs = redis_obj.bm25_search(question, index_name)

    if trace:
        trace.set("vector_hits", _summarize_hits(vector_docs, "vector_score"))
        trace.set("bm25_hits", _summarize_hits(bm25_docs, "score"))

    # ------------------------------------------------------------------
    # Stage 2: Reciprocal Rank Fusion
    # ------------------------------------------------------------------
    vector_ids = [d.id for d in vector_docs]
    bm25_ids = [d.id for d in bm25_docs]

    lists_to_fuse = [vector_ids]
    if cfg.HYBRID_SEARCH_ENABLED and bm25_ids:
        lists_to_fuse.append(bm25_ids)

    if len(lists_to_fuse) == 1:
        # Single-source: preserve the order, fabricate descending scores for trace
        fused = [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(vector_ids[:cfg.RRF_TOP_K])]
    else:
        if trace:
            with trace.timer("rrf"):
                fused = reciprocal_rank_fusion(lists_to_fuse, k=cfg.RRF_K, top_k=cfg.RRF_TOP_K)
        else:
            fused = reciprocal_rank_fusion(lists_to_fuse, k=cfg.RRF_K, top_k=cfg.RRF_TOP_K)

    if trace:
        trace.set("rrf_top", [{"id": d, "rrf_score": round(s, 5)} for d, s in fused[:10]])

    fused_ids = [d for d, _ in fused]
    if not fused_ids:
        if trace:
            trace.set("final_doc_ids", [])
        return []

    # Materialise the fused doc payloads (preserves RRF order)
    fused_docs = redis_obj.get_docs_by_ids(fused_ids)

    # ------------------------------------------------------------------
    # Stage 3: cross-encoder rerank
    # ------------------------------------------------------------------
    if cfg.RERANKER_ENABLED and fused_docs:
        pairs = [(d["id"], d.get("content", "")) for d in fused_docs]
        if trace:
            with trace.timer("rerank"):
                reranked = cross_encoder_rerank(question, pairs, top_n=cfg.RERANK_TOP_N)
        else:
            reranked = cross_encoder_rerank(question, pairs, top_n=cfg.RERANK_TOP_N)

        rerank_scores = {doc_id: score for doc_id, score in reranked}
        ordered_ids = [doc_id for doc_id, _ in reranked]
        id_to_doc = {d["id"]: d for d in fused_docs}
        final_docs = [id_to_doc[i] for i in ordered_ids if i in id_to_doc]

        if trace:
            trace.set("rerank_top", [
                {"id": doc_id, "rerank_score": round(rerank_scores[doc_id], 5)}
                for doc_id in ordered_ids
            ])
    else:
        final_docs = fused_docs[:cfg.RERANK_TOP_N]

    # ------------------------------------------------------------------
    # Stage 4 (optional): neighbor expansion
    # ------------------------------------------------------------------
    if cfg.INCLUDE_NEIGHBORS:
        final_docs = _attach_neighbors(redis_obj, final_docs)

    if trace:
        trace.set("final_doc_ids", [d["id"] for d in final_docs])

    current_app.logger.info(f"🎯 retrieve(): returning {len(final_docs)} docs")
    return final_docs
