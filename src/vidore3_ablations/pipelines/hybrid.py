"""Hybrid fusion pipelines combining text and visual retrieval."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from vidore3_ablations.evaluator import BasePipeline

from vidore3_ablations.pipelines.clip_visual import CLIPVisualPipeline
from vidore3_ablations.pipelines.dense_text import DenseTextPipeline
from vidore3_ablations.pipelines.utils import normalize_scores, reciprocal_rank_fusion, topk_scores


class _DualModalityMixin:
    """Shared indexing for hybrid pipelines."""

    text_model: str
    visual_model: str
    batch_size: int
    top_k: int

    def _build_modal_scores(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        text_pipe = DenseTextPipeline(
            model_name=self.text_model,
            batch_size=self.batch_size,
            top_k=len(self.corpus_ids),
        )
        visual_pipe = CLIPVisualPipeline(
            model_name=self.visual_model,
            batch_size=self.batch_size,
            top_k=len(self.corpus_ids),
        )
        text_pipe.corpus_ids = self.corpus_ids
        text_pipe.corpus_images = self.corpus_images
        text_pipe.corpus_texts = self.corpus_texts
        visual_pipe.corpus_ids = self.corpus_ids
        visual_pipe.corpus_images = self.corpus_images
        visual_pipe.corpus_texts = self.corpus_texts

        text_pipe.index(self.corpus_ids, self.corpus_images, self.corpus_texts)
        visual_pipe.index(self.corpus_ids, self.corpus_images, self.corpus_texts)

        text_results = text_pipe.search(query_ids, queries)
        visual_results = visual_pipe.search(query_ids, queries)

        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.corpus_ids)}
        text_matrix = np.zeros((len(query_ids), len(self.corpus_ids)), dtype=np.float32)
        visual_matrix = np.zeros((len(query_ids), len(self.corpus_ids)), dtype=np.float32)

        for q_idx, query_id in enumerate(query_ids):
            for doc_id, score in text_results[query_id].items():
                text_matrix[q_idx, id_to_idx[doc_id]] = score
            for doc_id, score in visual_results[query_id].items():
                visual_matrix[q_idx, id_to_idx[doc_id]] = score

        return text_matrix, visual_matrix


class HybridRRFPipeline(BasePipeline, _DualModalityMixin):
    """Fuse dense text OCR and CLIP visual rankings with reciprocal rank fusion."""

    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        visual_model: str = "openai/clip-vit-base-patch32",
        rrf_k: int = 60,
        batch_size: int = 32,
        top_k: int = 100,
    ):
        self.text_model = text_model
        self.visual_model = visual_model
        self.rrf_k = rrf_k
        self.batch_size = batch_size
        self.top_k = top_k

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        text_matrix, visual_matrix = self._build_modal_scores(query_ids, queries)
        results: Dict[str, Dict[str, float]] = {}

        for q_idx, query_id in enumerate(query_ids):
            text_ranked = [
                self.corpus_ids[i]
                for i in np.argsort(-text_matrix[q_idx])[: self.top_k]
            ]
            visual_ranked = [
                self.corpus_ids[i]
                for i in np.argsort(-visual_matrix[q_idx])[: self.top_k]
            ]
            fused = reciprocal_rank_fusion([text_ranked, visual_ranked], k=self.rrf_k)
            sorted_docs = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: self.top_k]
            results[query_id] = {doc_id: score for doc_id, score in sorted_docs}
        return results


class HybridWeightedPipeline(BasePipeline, _DualModalityMixin):
    """Fuse normalized dense text and CLIP visual cosine scores."""

    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        visual_model: str = "openai/clip-vit-base-patch32",
        text_weight: float = 0.5,
        batch_size: int = 32,
        top_k: int = 100,
    ):
        self.text_model = text_model
        self.visual_model = visual_model
        self.text_weight = text_weight
        self.batch_size = batch_size
        self.top_k = top_k

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        text_matrix, visual_matrix = self._build_modal_scores(query_ids, queries)
        results: Dict[str, Dict[str, float]] = {}
        visual_weight = 1.0 - self.text_weight

        for q_idx, query_id in enumerate(query_ids):
            text_norm = normalize_scores(text_matrix[q_idx])
            visual_norm = normalize_scores(visual_matrix[q_idx])
            fused = self.text_weight * text_norm + visual_weight * visual_norm
            results[query_id] = topk_scores(fused, self.corpus_ids, self.top_k)
        return results
