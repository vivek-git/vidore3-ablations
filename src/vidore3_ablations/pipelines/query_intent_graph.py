"""Query intent graph-aware retrieval pipelines."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from vidore3_ablations.evaluator import BasePipeline
from vidore3_ablations.pipelines.bm25 import tokenize
from vidore3_ablations.pipelines.utils import normalize_scores, topk_scores
from vidore3_ablations.query_intent import QueryIntentGraphBuilder, normalize_text


class QueryIntentGraphBM25Pipeline(BasePipeline):
    """
    BM25 retrieval reranked with query intent graph node and edge evidence.

    The base BM25 signal keeps the raw user wording in play. Intent nodes add
    weighted term evidence for concepts, attributes, identifiers, and values;
    graph edges add phrase/proximity evidence for linked concepts.
    """

    def __init__(
        self,
        top_k: int = 100,
        base_weight: float = 1.0,
        node_weight: float = 0.35,
        edge_weight: float = 0.25,
        include_graph_info: bool = True,
    ):
        self.top_k = top_k
        self.base_weight = base_weight
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.include_graph_info = include_graph_info
        self._bm25: BM25Okapi | None = None
        self._normalized_corpus_texts: List[str] = []

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        tokenized_corpus = [tokenize(text or "") for text in corpus_texts]
        self._normalized_corpus_texts = [normalize_text(text or "") for text in corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Dict[str, Dict[str, float]] | tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        assert self._bm25 is not None

        graphs = getattr(self, "query_intent_graphs", {})
        missing = [query_id for query_id in query_ids if query_id not in graphs]
        if missing:
            query_by_id = dict(zip(query_ids, queries))
            built = QueryIntentGraphBuilder().build_many(
                missing,
                [query_by_id[query_id] for query_id in missing],
            )
            self.set_query_intent_graphs({**graphs, **built})

        results: Dict[str, Dict[str, float]] = {}
        for query_id, query in zip(query_ids, queries):
            graph = self.query_intent_graphs[query_id]
            base_scores = self._bm25.get_scores(tokenize(query))
            node_scores = self._score_nodes(graph)
            edge_scores = self._score_edges(graph)
            scores = (
                self.base_weight * normalize_scores(base_scores)
                + self.node_weight * normalize_scores(node_scores)
                + self.edge_weight * normalize_scores(edge_scores)
            )
            results[query_id] = topk_scores(scores, self.corpus_ids, self.top_k)

        if not self.include_graph_info:
            return results

        return results, {
            "query_intent_graphs": {
                query_id: self.query_intent_graphs[query_id].to_dict()
                for query_id in query_ids
            },
            "query_intent_scoring": {
                "base_weight": self.base_weight,
                "node_weight": self.node_weight,
                "edge_weight": self.edge_weight,
            },
        }

    def _score_nodes(self, graph) -> np.ndarray:
        assert self._bm25 is not None
        scores = np.zeros(len(self.corpus_ids), dtype=np.float32)
        for node in graph.nodes:
            node_tokens = tokenize(node.text)
            if not node_tokens:
                continue
            scores += node.weight * self._bm25.get_scores(node_tokens)
        return scores

    def _score_edges(self, graph) -> np.ndarray:
        assert self._bm25 is not None
        scores = np.zeros(len(self.corpus_ids), dtype=np.float32)
        for edge in graph.edges:
            edge_tokens = tokenize(edge.text)
            if not edge_tokens:
                continue
            scores += edge.weight * self._bm25.get_scores(edge_tokens)

            normalized_edge = normalize_text(edge.text)
            if not normalized_edge:
                continue
            phrase_hits = np.array(
                [
                    1.0 if normalized_edge in normalized_text else 0.0
                    for normalized_text in self._normalized_corpus_texts
                ],
                dtype=np.float32,
            )
            scores += edge.weight * phrase_hits
        return scores
