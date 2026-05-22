"""CLIP-based visual and text-on-OCR retrieval pipelines."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from vidore3_ablations.device_utils import DevicePreference, empty_cuda_cache, get_torch_device
from vidore3_ablations.evaluator import BasePipeline

from vidore3_ablations.pipelines.utils import batch_iter, topk_scores


def _scale_image(image: Image.Image, scale: float) -> Image.Image:
    if scale >= 0.999:
        return image
    width, height = image.size
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


class CLIPVisualPipeline(BasePipeline):
    """Dense visual retrieval: cosine similarity between query text and page images."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 8,
        top_k: int = 100,
        image_scale: float = 1.0,
        blank_corpus_text: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.image_scale = image_scale
        self.blank_corpus_text = blank_corpus_text
        self._device = get_torch_device()
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._image_embeddings: np.ndarray | None = None

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        if self.blank_corpus_text:
            self.corpus_texts = [""] * len(corpus_texts)

        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.eval()
        use_autocast = self._device.type == "cuda"

        embeddings: List[np.ndarray] = []
        for _, batch_images in batch_iter(corpus_images, self.batch_size):
            images = [_scale_image(img.convert("RGB"), self.image_scale) for img in batch_images]
            inputs = self._processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                with torch.autocast(device_type=self._device.type, enabled=use_autocast):
                    features = self._model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            embeddings.append(features.float().cpu().numpy())
            empty_cuda_cache()
        self._image_embeddings = np.vstack(embeddings)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        assert self._model is not None and self._processor is not None
        assert self._image_embeddings is not None
        results: Dict[str, Dict[str, float]] = {}
        use_autocast = self._device.type == "cuda"

        for start, batch_queries in batch_iter(queries, self.batch_size):
            batch_ids = query_ids[start : start + len(batch_queries)]
            inputs = self._processor(text=batch_queries, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                with torch.autocast(device_type=self._device.type, enabled=use_autocast):
                    features = self._model.get_text_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            query_embeddings = features.float().cpu().numpy()
            sims = query_embeddings @ self._image_embeddings.T
            for offset, query_id in enumerate(batch_ids):
                results[query_id] = topk_scores(sims[offset], self.corpus_ids, self.top_k)
            empty_cuda_cache()
        return results


class CLIPTextOnOCRPipeline(BasePipeline):
    """Ablation: CLIP text encoder on OCR markdown (no page images at retrieval time)."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
        top_k: int = 100,
        device: DevicePreference = "auto",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self._device = get_torch_device(device)
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._doc_embeddings: np.ndarray | None = None

    def index(
        self,
        corpus_ids: List[str],
        corpus_images,
        corpus_texts: List[str],
        dataset_name=None,
    ) -> None:
        super().index(corpus_ids, corpus_images, corpus_texts, dataset_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.eval()
        use_autocast = self._device.type == "cuda"

        texts = [text or "" for text in corpus_texts]
        embeddings: List[np.ndarray] = []
        for _, batch in batch_iter(texts, self.batch_size):
            inputs = self._processor(text=list(batch), return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                with torch.autocast(device_type=self._device.type, enabled=use_autocast):
                    features = self._model.get_text_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            embeddings.append(features.float().cpu().numpy())
            empty_cuda_cache()
        self._doc_embeddings = np.vstack(embeddings)

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        assert self._model is not None and self._processor is not None
        assert self._doc_embeddings is not None
        results: Dict[str, Dict[str, float]] = {}
        use_autocast = self._device.type == "cuda"

        for start, batch_queries in batch_iter(queries, self.batch_size):
            batch_ids = query_ids[start : start + len(batch_queries)]
            inputs = self._processor(text=batch_queries, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                with torch.autocast(device_type=self._device.type, enabled=use_autocast):
                    features = self._model.get_text_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            query_embeddings = features.float().cpu().numpy()
            sims = query_embeddings @ self._doc_embeddings.T
            for offset, query_id in enumerate(batch_ids):
                results[query_id] = topk_scores(sims[offset], self.corpus_ids, self.top_k)
            empty_cuda_cache()
        return results
