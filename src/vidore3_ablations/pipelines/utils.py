"""Shared helpers for retrieval pipelines."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def topk_scores(
    scores: np.ndarray,
    corpus_ids: Sequence[str],
    k: int,
) -> Dict[str, float]:
    """Return top-k corpus_id -> score mapping from a score vector."""
    k = min(k, len(scores))
    if k == 0:
        return {}
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return {corpus_ids[i]: float(scores[i]) for i in top_indices}


def batch_iter(items: Sequence, batch_size: int) -> Iterable[Tuple[int, Sequence]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> Dict[str, float]:
    """Fuse multiple ranked doc-id lists with RRF."""
    fused: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score - min_score < 1e-12:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)
