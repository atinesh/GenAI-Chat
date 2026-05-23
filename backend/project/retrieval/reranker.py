"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import os
from typing import List, Tuple
from huggingface_hub import InferenceClient
import config as cfg

_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN environment variable is not set")
        _client = InferenceClient(provider=cfg.RERANKER_PROVIDER, api_key=token)
    return _client


def _pair_to_input(query: str, passage: str) -> str:
    """
    bge-m3 / XLM-RoBERTa expects pair inputs joined by `</s></s>`.
    """
    return f"{query}</s></s>{passage}"


def _extract_score(result) -> float:
    """
    Normalize the HF text-classification response into a single float score.

    HF Inference may return either:
        - a list of {label, score} dicts (top-k labels)
        - a list of lists (one inner list per input)
    For a binary cross-encoder we take the max score across labels (the model's
    "relevant" label is what carries signal).
    """
    if not result:
        return 0.0

    # Flatten one level of nesting if present
    if isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]

    try:
        scores = [float(item.get("score", 0.0)) for item in result if isinstance(item, dict)]
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0


def rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    top_n: int | None = None,
) -> List[Tuple[str, float]]:
    """
    Rerank candidates by cross-encoder relevance to `query`.

    Args:
        query:      user query (or rewritten query).
        candidates: list of (doc_id, passage_text) tuples.
        top_n:      keep only top-N after rerank (default: cfg.RERANK_TOP_N).

    Returns:
        list of (doc_id, rerank_score) sorted desc, length <= top_n.
        On total failure, returns the input order with score=0.0.
    """
    if not candidates:
        return []
    top_n = top_n or cfg.RERANK_TOP_N

    try:
        client = _get_client()
    except Exception:
        # No token / client error → preserve input order
        return [(doc_id, 0.0) for doc_id, _ in candidates[:top_n]]

    scored: List[Tuple[str, float]] = []
    for doc_id, passage in candidates:
        try:
            result = client.text_classification(
                _pair_to_input(query, passage),
                model=cfg.RERANKER_MODEL,
            )
            scored.append((doc_id, _extract_score(result)))
        except Exception:
            # Per-candidate failure → score 0, keep going
            scored.append((doc_id, 0.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
