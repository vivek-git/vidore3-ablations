"""Per-query region grounding evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from vidore3_ablations.metrics.definitions import (
    DEFAULT_F1_SUCCESS_THRESHOLD,
    DEFAULT_IOU_SUCCESS_THRESHOLD,
)
from vidore3_ablations.metrics.grounding import (
    Box,
    best_match_overlap,
    full_page_box,
    greedy_box_map,
    group_boxes_by_annotator,
    parse_boxes,
)


def evaluate_grounding(
    run: Dict[str, Dict[str, float]],
    query_ids: List[str],
    corpus_ids: List[str],
    corpus_images: List[Any],
    qrels: Dict[str, Dict[str, int]],
    qrels_boxes: Mapping[str, Mapping[str, Sequence[dict]]],
    predictions: Mapping[str, Mapping[str, Sequence[Box]]],
    iou_success_threshold: float = DEFAULT_IOU_SUCCESS_THRESHOLD,
    f1_success_threshold: float = DEFAULT_F1_SUCCESS_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    corpus_id_to_image = {cid: img for cid, img in zip(corpus_ids, corpus_images)}
    per_query: Dict[str, Dict[str, float]] = {}

    for query_id in query_ids:
        scores = run.get(query_id, {})
        ranked = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        relevant = set(qrels.get(query_id, {}))
        annotated_pages = set(qrels_boxes.get(query_id, {}))

        top1 = ranked[0] if ranked else None
        top5 = ranked[:5]
        top10 = ranked[:10]

        overlap_at_1 = None
        box_metrics_at_1 = (0.0, 0.0)
        if top1 and top1 in qrels_boxes.get(query_id, {}):
            image = corpus_id_to_image[top1]
            gt_by_annotator = group_boxes_by_annotator(qrels_boxes[query_id][top1])
            pred_boxes = list(predictions.get(query_id, {}).get(top1, [full_page_box(image.width, image.height)]))
            overlap_at_1 = best_match_overlap(pred_boxes, gt_by_annotator, image.width, image.height)
            gt_flat = parse_boxes(qrels_boxes[query_id][top1])
            box_metrics_at_1 = greedy_box_map(pred_boxes, gt_flat, iou_threshold=0.5)

        best_iou_at_5 = 0.0
        best_f1_at_5 = 0.0
        for corpus_id in top5:
            if corpus_id not in qrels_boxes.get(query_id, {}):
                continue
            image = corpus_id_to_image[corpus_id]
            gt_by_annotator = group_boxes_by_annotator(qrels_boxes[query_id][corpus_id])
            pred_boxes = list(predictions.get(query_id, {}).get(corpus_id, [full_page_box(image.width, image.height)]))
            overlap = best_match_overlap(pred_boxes, gt_by_annotator, image.width, image.height)
            best_iou_at_5 = max(best_iou_at_5, overlap.iou)
            best_f1_at_5 = max(best_f1_at_5, overlap.f1)

        per_query[query_id] = {
            "grounding_iou_at_1": overlap_at_1.iou if overlap_at_1 else 0.0,
            "grounding_f1_at_1": overlap_at_1.f1 if overlap_at_1 else 0.0,
            "grounding_precision_at_1": overlap_at_1.precision if overlap_at_1 else 0.0,
            "grounding_recall_at_1": overlap_at_1.recall if overlap_at_1 else 0.0,
            "grounding_iou_at_5": best_iou_at_5,
            "grounding_f1_at_5": best_f1_at_5,
            "box_precision_at_0.5": box_metrics_at_1[0],
            "box_recall_at_0.5": box_metrics_at_1[1],
            "grounded_success_iou50_at_1": float(
                top1 in relevant
                and overlap_at_1 is not None
                and overlap_at_1.iou >= iou_success_threshold
            ),
            "grounded_success_f1_at_1": float(
                top1 in relevant
                and overlap_at_1 is not None
                and overlap_at_1.f1 >= f1_success_threshold
            ),
            "localization_recall_at_5": float(bool(annotated_pages & set(top5))),
            "localization_recall_at_10": float(bool(annotated_pages & set(top10))),
            "retrieval_and_grounding_at_1": float(
                top1 in relevant and overlap_at_1 is not None and overlap_at_1.f1 > 0.0
            ),
        }

    return per_query
