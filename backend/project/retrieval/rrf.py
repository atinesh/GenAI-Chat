"""
Reciprocal Rank Fusion (RRF).
RRF fuses results from multiple ranked lists without needing comparable scores.
"""

from typing import Dict, Iterable, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Iterable[List[str]],
    k: int = 60,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Args:
        ranked_lists: iterable of ranked lists of doc-IDs (best first).
        k:           RRF constant (60 is the canonical value).
        top_k:       number of fused results to return.

    Returns:
        list of (doc_id, fused_score) sorted desc, length <= top_k.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):  # 1-based rank
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]
