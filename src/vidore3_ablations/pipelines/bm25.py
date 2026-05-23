"""BM25 sparse text retrieval on OCR markdown."""

from __future__ import annotations

import re
from typing import Dict, List

from rank_bm25 import BM25Okapi
from vidore3_ablations.pipelines.base import BasePipeline

from vidore3_ablations.pipelines.utils import topk_scores


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25TextPipeline(BasePipeline):
    """Lexical retrieval using OCR-extracted markdown only."""

    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: List[List[str]] = []

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        self._tokenized_corpus = [tokenize(text or "") for text in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        assert self._bm25 is not None
        results: Dict[str, Dict[str, float]] = {}
        for query_id, query in zip(query_ids, queries):
            scores = self._bm25.get_scores(tokenize(query))
            results[query_id] = topk_scores(scores, self.corpus_ids, self.top_k)
        return results
