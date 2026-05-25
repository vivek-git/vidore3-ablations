"""Retrieval pipeline implementations for ablation experiments."""

from vidore3_ablations.pipelines.bm25 import BM25TextPipeline
from vidore3_ablations.pipelines.clip_visual import CLIPTextOnOCRPipeline, CLIPVisualPipeline
from vidore3_ablations.pipelines.colpali import ColPaliLateInteractionPipeline
from vidore3_ablations.pipelines.dense_text import DenseTextPipeline
from vidore3_ablations.pipelines.hybrid import HybridRRFPipeline, HybridWeightedPipeline
from vidore3_ablations.pipelines.query_intent_graph import QueryIntentGraphBM25Pipeline
from vidore3_ablations.pipelines.random import RandomPipeline
from vidore3_ablations.pipelines.registry import PIPELINE_REGISTRY, build_pipeline

__all__ = [
    "BM25TextPipeline",
    "CLIPTextOnOCRPipeline",
    "CLIPVisualPipeline",
    "ColPaliLateInteractionPipeline",
    "DenseTextPipeline",
    "HybridRRFPipeline",
    "HybridWeightedPipeline",
    "QueryIntentGraphBM25Pipeline",
    "RandomPipeline",
    "PIPELINE_REGISTRY",
    "build_pipeline",
]
