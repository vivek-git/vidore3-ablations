"""Region grounding metrics for ViDoRe v3 visual document retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 inclusive pixel coordinates


@dataclass(frozen=True)
class RegionOverlap:
    iou: float
    f1: float
    precision: float
    recall: float
    intersection: int
    union: int
    pred_area: int
    gt_area: int


def parse_boxes(raw_boxes: Sequence[Mapping[str, int]]) -> List[Box]:
    boxes: List[Box] = []
    for item in raw_boxes:
        x1, y1, x2, y2 = int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        boxes.append((x1, y1, x2, y2))
    return boxes


def group_boxes_by_annotator(raw_boxes: Sequence[Mapping[str, int]]) -> Dict[int, List[Box]]:
    grouped: Dict[int, List[Box]] = {}
    for item in raw_boxes:
        annotator = int(item.get("annotator", 0))
        grouped.setdefault(annotator, []).append(parse_boxes([item])[0])
    return grouped


def full_page_box(width: int, height: int) -> Box:
    return (0, 0, max(width - 1, 0), max(height - 1, 0))


def _zone_mask(width: int, height: int, boxes: Sequence[Box]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def region_overlap(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], width: int, height: int) -> RegionOverlap:
    """Pixel-level overlap between merged prediction and ground-truth zones."""
    if width <= 0 or height <= 0:
        return RegionOverlap(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

    pred_mask = _zone_mask(width, height, pred_boxes)
    gt_mask = _zone_mask(width, height, gt_boxes)
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    pred_area = int(pred_mask.sum())
    gt_area = int(gt_mask.sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())

    iou = intersection / union if union else 0.0
    precision = intersection / pred_area if pred_area else 0.0
    recall = intersection / gt_area if gt_area else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return RegionOverlap(iou, f1, precision, recall, intersection, union, pred_area, gt_area)


def best_match_overlap(
    pred_boxes: Sequence[Box],
    gt_by_annotator: Mapping[int, Sequence[Box]],
    width: int,
    height: int,
) -> RegionOverlap:
    """Best-match score across human annotators (ViDoRe v3 grounding protocol)."""
    if not gt_by_annotator:
        return RegionOverlap(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

    overlaps = [
        region_overlap(pred_boxes, gt_boxes, width, height)
        for gt_boxes in gt_by_annotator.values()
        if gt_boxes
    ]
    if not overlaps:
        return RegionOverlap(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    return max(overlaps, key=lambda item: item.f1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def greedy_box_map(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], iou_threshold: float = 0.5) -> Tuple[float, float]:
    """Greedy box-level mean average precision proxy and recall at an IoU threshold."""
    if not gt_boxes:
        return 0.0, 0.0
    if not pred_boxes:
        return 0.0, 0.0

    matched_gt = set()
    true_positives = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            true_positives += 1
            matched_gt.add(best_idx)

    precision = true_positives / len(pred_boxes)
    recall = true_positives / len(gt_boxes)
    return precision, recall


def mean_metric(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
