"""Jina Embeddings v5 Omni Nano multimodal retrieval pipeline."""

from __future__ import annotations

from typing import Dict, List, Literal, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm
from vidore3_ablations.evaluator import BasePipeline

from vidore3_ablations.pipelines.utils import batch_iter, topk_scores

DocumentMode = Literal["image", "ocr_text", "image_text"]


class JinaV5OmniNanoPipeline(BasePipeline):
    """
    Text-to-page retrieval with Jina v5 Omni Nano retrieval embeddings.

    The default configuration benchmarks the multimodal path: text queries are
    embedded as retrieval queries and rendered page images are embedded as
    retrieval documents in the same vector space.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v5-omni-nano-retrieval",
        batch_size: int = 8,
        top_k: int = 100,
        document_mode: DocumentMode = "image",
        modality: str = "vision",
        truncate_dim: int | None = None,
        device: str | None = None,
        show_progress: bool = True,
    ):
        if document_mode not in {"image", "ocr_text", "image_text"}:
            raise ValueError(
                "document_mode must be one of: image, ocr_text, image_text"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.document_mode = document_mode
        self.modality = modality
        self.truncate_dim = truncate_dim
        self.device = device
        self.show_progress = show_progress

        self._model = None
        self._document_embeddings: np.ndarray | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to run the Jina v5 Omni Nano pipeline."
            ) from exc

        kwargs = {
            "trust_remote_code": True,
            "model_kwargs": {"modality": self.modality},
        }
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)

    def _encode(
        self,
        inputs: Sequence,
        side: Literal["query", "document"],
    ) -> np.ndarray:
        assert self._model is not None

        if side == "query":
            encode_fn = getattr(self._model, "encode_query")
        else:
            encode_fn = getattr(self._model, "encode_document")

        kwargs = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self.truncate_dim is not None:
            kwargs["truncate_dim"] = self.truncate_dim

        embeddings = encode_fn(list(inputs), **kwargs)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    def _prepare_documents(
        self,
        corpus_images: Sequence[Image.Image],
        corpus_texts: Sequence[str],
    ) -> List:
        if self.document_mode == "image":
            return [image.convert("RGB") for image in corpus_images]
        if self.document_mode == "ocr_text":
            return [text or "" for text in corpus_texts]
        return [
            (text or "", image.convert("RGB"))
            for image, text in zip(corpus_images, corpus_texts)
        ]

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        self._load_model()

        documents = self._prepare_documents(corpus_images, corpus_texts)
        batches = list(batch_iter(documents, self.batch_size))
        iterator = (
            tqdm(batches, desc="Jina v5 Omni index", leave=False)
            if self.show_progress
            else batches
        )

        embeddings: List[np.ndarray] = []
        for _, batch in iterator:
            embeddings.append(self._encode(batch, side="document"))
        self._document_embeddings = np.vstack(embeddings)

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Dict[str, Dict[str, float]]:
        assert self._document_embeddings is not None, "Call index() before search()"

        batches = list(batch_iter(queries, self.batch_size))
        iterator = (
            tqdm(batches, desc="Jina v5 Omni queries", leave=False)
            if self.show_progress
            else batches
        )
        results: Dict[str, Dict[str, float]] = {}

        for start, batch_queries in iterator:
            batch_ids = query_ids[start : start + len(batch_queries)]
            query_embeddings = self._encode(batch_queries, side="query")
            sims = query_embeddings @ self._document_embeddings.T
            for offset, query_id in enumerate(batch_ids):
                results[query_id] = topk_scores(
                    sims[offset],
                    self.corpus_ids,
                    self.top_k,
                )

        return results
