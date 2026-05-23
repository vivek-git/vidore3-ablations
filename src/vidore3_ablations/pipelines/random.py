"""Random baseline pipeline."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from vidore3_ablations.pipelines.base import BasePipeline


class RandomPipeline(BasePipeline):
    def __init__(self, seed: int = 42, top_k: int = 100):
        self.seed = seed
        self.top_k = top_k
        self._rng = random.Random(seed)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for query_id in query_ids:
            shuffled = list(self.corpus_ids)
            self._rng.shuffle(shuffled)
            top = shuffled[: self.top_k]
            results[query_id] = {doc_id: float(len(top) - rank) for rank, doc_id in enumerate(top)}
        return results
