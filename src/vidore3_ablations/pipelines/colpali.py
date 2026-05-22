"""ColPali late-interaction (MaxSim) visual document retrieval."""

from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from vidore3_ablations.device_utils import (
    DevicePreference,
    detect_vram_profile,
    empty_cuda_cache,
    get_inference_dtype,
    get_torch_device,
    resolve_score_device,
)
from vidore3_ablations.evaluator import BasePipeline

from vidore3_ablations.pipelines.utils import batch_iter, topk_scores


class ColPaliLateInteractionPipeline(BasePipeline):
    """
    ColBERT-style late interaction over page-image patch embeddings.

    Tuned for 12 GB GPUs: encodes one page at a time and runs MaxSim scoring on CPU
    by default to avoid VRAM spikes on large corpora.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        batch_size: int = 1,
        score_batch_size: int = 32,
        top_k: int = 100,
        device: DevicePreference = "auto",
        score_device: DevicePreference = "auto",
        max_gpu_memory: str | None = None,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.score_batch_size = score_batch_size
        self.top_k = top_k
        self.show_progress = show_progress
        self.max_gpu_memory = max_gpu_memory
        self._device = get_torch_device(device)
        self._score_device_preference: DevicePreference = score_device
        self._score_device = torch.device("cpu")
        self._dtype = get_inference_dtype(self._device)
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

        load_kwargs: Dict = {
            "torch_dtype": self._dtype,
            "low_cpu_mem_usage": True,
        }
        if self.max_gpu_memory and self._device.type == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {0: self.max_gpu_memory, "cpu": "48GiB"}
        else:
            load_kwargs["device_map"] = str(self._device)

        self._model = ColPali.from_pretrained(self.model_name, **load_kwargs).eval()
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)

    def _input_device(self) -> torch.device:
        assert self._model is not None
        return next(self._model.parameters()).device

    def _encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        assert self._model is not None and self._processor is not None
        embeddings: List[torch.Tensor] = []
        batches = list(batch_iter(images, self.batch_size))
        iterator = tqdm(batches, desc="ColPali index", leave=False) if self.show_progress else batches

        for _, batch_images in iterator:
            rgb_images = [img.convert("RGB") for img in batch_images]
            batch = self._processor.process_images(rgb_images)
            batch = {key: value.to(self._input_device()) for key, value in batch.items()}
            with torch.inference_mode():
                batch_embeddings = self._model(**batch)
            embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
            empty_cuda_cache()
        return embeddings

    def _encode_queries(self, queries: List[str]) -> List[torch.Tensor]:
        assert self._model is not None and self._processor is not None
        embeddings: List[torch.Tensor] = []
        batches = list(batch_iter(queries, self.batch_size))
        iterator = tqdm(batches, desc="ColPali queries", leave=False) if self.show_progress else batches

        for _, batch_queries in iterator:
            batch = self._processor.process_queries(list(batch_queries))
            batch = {key: value.to(self._input_device()) for key, value in batch.items()}
            with torch.inference_mode():
                batch_embeddings = self._model(**batch)
            embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
            empty_cuda_cache()
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
        profile = "ultra_8gb" if self.max_gpu_memory else detect_vram_profile()
        self._score_device = resolve_score_device(
            self._device,
            self._score_device_preference,
            len(corpus_images),
            vram_profile=profile,
        )
        self._passage_embeddings = self._encode_images(corpus_images)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        assert self._processor is not None
        assert self._passage_embeddings, "Call index() before search()"

        query_embeddings = self._encode_queries(queries)
        scores = self._processor.score_multi_vector(
            query_embeddings,
            self._passage_embeddings,
            batch_size=self.score_batch_size,
            device=self._score_device,
        )
        empty_cuda_cache()

        results: Dict[str, Dict[str, float]] = {}
        for q_idx, query_id in enumerate(query_ids):
            results[query_id] = topk_scores(
                scores[q_idx].numpy(),
                self.corpus_ids,
                self.top_k,
            )
        return results
