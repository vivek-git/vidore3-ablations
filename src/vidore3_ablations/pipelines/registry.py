"""Pipeline registry and factory."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from vidore3_ablations.evaluator import BasePipeline

from vidore3_ablations.pipelines.bm25 import BM25TextPipeline
from vidore3_ablations.pipelines.clip_visual import CLIPTextOnOCRPipeline, CLIPVisualPipeline
from vidore3_ablations.pipelines.colpali import ColPaliLateInteractionPipeline
from vidore3_ablations.pipelines.dense_text import DenseTextPipeline
from vidore3_ablations.pipelines.hybrid import HybridRRFPipeline, HybridWeightedPipeline
from vidore3_ablations.pipelines.jina_omni import JinaV5OmniNanoPipeline
from vidore3_ablations.pipelines.random import RandomPipeline

PipelineFactory = Callable[..., BasePipeline]

PIPELINE_REGISTRY: Dict[str, Type[BasePipeline]] = {
    "random": RandomPipeline,
    "bm25_text": BM25TextPipeline,
    "dense_text": DenseTextPipeline,
    "clip_visual": CLIPVisualPipeline,
    "clip_text_on_ocr": CLIPTextOnOCRPipeline,
    "hybrid_rrf": HybridRRFPipeline,
    "hybrid_weighted": HybridWeightedPipeline,
    "colpali_late_interaction": ColPaliLateInteractionPipeline,
    "jina_v5_omni_nano": JinaV5OmniNanoPipeline,
}


def build_pipeline(name: str, **kwargs: Any) -> BasePipeline:
    if name not in PIPELINE_REGISTRY:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")
    return PIPELINE_REGISTRY[name](**kwargs)
