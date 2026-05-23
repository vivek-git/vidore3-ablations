"""Metric name catalogs and descriptions."""

from __future__ import annotations

RETRIEVAL_METRICS = [
    "ndcg_cut_5",
    "ndcg_cut_10",
    "recall_5",
    "recall_10",
    "map",
]

GROUNDING_METRICS = [
    "grounding_iou_at_1",
    "grounding_f1_at_1",
    "grounding_precision_at_1",
    "grounding_recall_at_1",
    "grounding_iou_at_5",
    "grounding_f1_at_5",
    "box_precision_at_0.5",
    "box_recall_at_0.5",
]

END_TO_END_METRICS = [
    "grounded_success_iou50_at_1",
    "grounded_success_f1_at_1",
    "localization_recall_at_5",
    "localization_recall_at_10",
    "retrieval_and_grounding_at_1",
]

ALL_GROUNDING_METRICS = GROUNDING_METRICS + END_TO_END_METRICS

DEFAULT_IOU_SUCCESS_THRESHOLD = 0.5
DEFAULT_F1_SUCCESS_THRESHOLD = 0.5

METRIC_DESCRIPTIONS = {
    "ndcg_cut_10": "Normalized discounted cumulative gain at rank 10 for page retrieval.",
    "recall_10": "Fraction of queries with at least one relevant page in the top 10.",
    "map": "Mean average precision over graded page relevance judgments.",
    "grounding_iou_at_1": "Best-match pixel IoU between predicted and human zones on the rank-1 page.",
    "grounding_f1_at_1": "Best-match pixel F1 (Dice) between predicted and human zones on the rank-1 page.",
    "grounding_precision_at_1": "Pixel precision of the predicted evidence zone on the rank-1 page.",
    "grounding_recall_at_1": "Pixel recall of the predicted evidence zone on the rank-1 page.",
    "grounding_iou_at_5": "Max best-match IoU over the top-5 retrieved pages.",
    "grounding_f1_at_5": "Max best-match F1 over the top-5 retrieved pages.",
    "box_precision_at_0.5": "Greedy box-level precision at IoU threshold 0.5 on rank-1 page.",
    "box_recall_at_0.5": "Greedy box-level recall at IoU threshold 0.5 on rank-1 page.",
    "grounded_success_iou50_at_1": "Rank-1 page is relevant and zone IoU >= 0.5 against any annotator.",
    "grounded_success_f1_at_1": "Rank-1 page is relevant and zone F1 >= 0.5 against any annotator.",
    "localization_recall_at_5": "Share of bbox-annotated queries with any annotated page in top-5.",
    "localization_recall_at_10": "Share of bbox-annotated queries with any annotated page in top-10.",
    "retrieval_and_grounding_at_1": "Rank-1 page is relevant and achieves non-zero grounding F1.",
}
