"""Metric geometry and overlap calculations."""

from vidore3_ablations.metrics.grounding import (
    Box,
    RegionOverlap,
    best_match_overlap,
    box_iou,
    full_page_box,
    greedy_box_map,
    group_boxes_by_annotator,
    mean_metric,
    parse_boxes,
    region_overlap,
)

__all__ = [
    "Box",
    "RegionOverlap",
    "best_match_overlap",
    "box_iou",
    "full_page_box",
    "greedy_box_map",
    "group_boxes_by_annotator",
    "mean_metric",
    "parse_boxes",
    "region_overlap",
]
