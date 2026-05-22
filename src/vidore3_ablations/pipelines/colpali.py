"""ColPali late-interaction (MaxSim) visual document retrieval."""

from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from vidore3_ablations.evaluator import BasePipeline

from vidore3_ablations.pipelines.utils import batch_iter, topk_scores


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


class ColPaliLateInteractionPipeline(BasePipeline):
    """
    ColBERT-style late interaction over page-image patch embeddings.

    Indexes multi-vector document representations from page images, then scores
    queries with MaxSim (sum of max token-patch similarities).
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        batch_size: int = 4,
        score_batch_size: int = 128,
        top_k: int = 100,
        device: str = "auto",
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.score_batch_size = score_batch_size
        self.top_k = top_k
        self.show_progress = show_progress

        if device == "auto":
            self._device = _get_device()
        else:
            self._device = torch.device(device)

        self._dtype = _get_dtype(self._device)
        self._model = None
        self._processor = None
        self._passage_embeddings: List[torch.Tensor] = []

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as exc:
            raise ImportError(
                'Install colpali-engine with: pip install "colpali-engine>=0.3.1"'
            ) from exc

        self._model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            device_map=str(self._device),
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)

    def _encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        assert self._model is not None and self._processor is not None
        embeddings: List[torch.Tensor] = []
        batches = list(batch_iter(images, self.batch_size))
        iterator = tqdm(batches, desc="ColPali index", leave=False) if self.show_progress else batches

        for _, batch_images in iterator:
            rgb_images = [img.convert("RGB") for img in batch_images]
            batch = self._processor.process_images(rgb_images).to(self._device)
            with torch.no_grad():
                batch_embeddings = self._model(**batch)
            embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
        return embeddings

    def _encode_queries(self, queries: List[str]) -> List[torch.Tensor]:
        assert self._model is not None and self._processor is not None
        embeddings: List[torch.Tensor] = []
        batches = list(batch_iter(queries, self.batch_size))
        iterator = tqdm(batches, desc="ColPali queries", leave=False) if self.show_progress else batches

        for _, batch_queries in iterator:
            batch = self._processor.process_queries(list(batch_queries)).to(self._device)
            with torch.no_grad():
                batch_embeddings = self._model(**batch)
            embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
        return embeddings

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        self._load_model()
        self._passage_embeddings = self._encode_images(corpus_images)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        assert self._processor is not None
        assert self._passage_embeddings, "Call index() before search()"

        query_embeddings = self._encode_queries(queries)
        scores = self._processor.score_multi_vector(
            query_embeddings,
            self._passage_embeddings,
            batch_size=self.score_batch_size,
            device=self._device,
        )

        results: Dict[str, Dict[str, float]] = {}
        for q_idx, query_id in enumerate(query_ids):
            results[query_id] = topk_scores(
                scores[q_idx].numpy(),
                self.corpus_ids,
                self.top_k,
            )
        return results
