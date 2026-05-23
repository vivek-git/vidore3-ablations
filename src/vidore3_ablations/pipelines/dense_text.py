"""Dense bi-encoder text retrieval on OCR markdown."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from vidore3_ablations.hardware.device import DevicePreference, get_torch_device
from vidore3_ablations.pipelines.base import BasePipeline

from vidore3_ablations.pipelines.utils import batch_iter, topk_scores


class DenseTextPipeline(BasePipeline):
    """Text-only dense retrieval using sentence-transformers on OCR markdown."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        top_k: int = 100,
        device: DevicePreference = "auto",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self._device = get_torch_device(device)
        self._model: SentenceTransformer | None = None
        self._corpus_embeddings: np.ndarray | None = None

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        self._model = SentenceTransformer(self.model_name, device=str(self._device))
        texts = [text or "" for text in corpus_texts]
        embeddings = []
        for _, batch in batch_iter(texts, self.batch_size):
            embeddings.append(
                self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
            )
        self._corpus_embeddings = np.vstack(embeddings)
        norms = np.linalg.norm(self._corpus_embeddings, axis=1, keepdims=True)
        self._corpus_embeddings = self._corpus_embeddings / np.clip(norms, 1e-12, None)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        assert self._model is not None and self._corpus_embeddings is not None
        results: Dict[str, Dict[str, float]] = {}
        for start, batch_queries in batch_iter(queries, self.batch_size):
            batch_ids = query_ids[start : start + len(batch_queries)]
            query_embeddings = self._model.encode(
                batch_queries,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / np.clip(norms, 1e-12, None)
            sims = query_embeddings @ self._corpus_embeddings.T
            for offset, query_id in enumerate(batch_ids):
                results[query_id] = topk_scores(sims[offset], self.corpus_ids, self.top_k)
        return results
