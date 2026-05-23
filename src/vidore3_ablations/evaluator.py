"""Backward-compatible re-exports. Prefer vidore3_ablations.eval and pipelines.base."""

from vidore3_ablations.eval import aggregate_results, evaluate_grounding, evaluate_retrieval
from vidore3_ablations.pipelines.base import BasePipeline

__all__ = [
    "BasePipeline",
    "aggregate_results",
    "evaluate_grounding",
    "evaluate_retrieval",
]
