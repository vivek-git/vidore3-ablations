"""Convert ViDoRe pixel boxes to FiftyOne detection format."""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

Box = Tuple[int, int, int, int]


def pixel_box_to_normalized(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> List[float]:
    """Convert inclusive pixel xyxy to FiftyOne [nx, ny, nw, nh] in [0, 1]."""
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    nx = max(0.0, min(1.0, x1 / width))
    ny = max(0.0, min(1.0, y1 / height))
    nw = max(0.0, min(1.0, (x2 - x1 + 1) / width))
    nh = max(0.0, min(1.0, (y2 - y1 + 1) / height))
    return [nx, ny, nw, nh]


def boxes_to_fiftyone_detections(
    raw_boxes: Sequence[Mapping[str, int]],
    width: int,
    height: int,
    label_prefix: str = "evidence",
):
    """Build fo.Detections from ViDoRe qrel bounding box dicts."""
    import fiftyone as fo

    detections = []
    for item in raw_boxes:
        annotator = int(item.get("annotator", 0))
        label = f"{label_prefix}_a{annotator}"
        bbox = pixel_box_to_normalized(
            int(item["x1"]),
            int(item["y1"]),
            int(item["x2"]),
            int(item["y2"]),
            width,
            height,
        )
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=bbox,
            )
        )
    return fo.Detections(detections=detections)
