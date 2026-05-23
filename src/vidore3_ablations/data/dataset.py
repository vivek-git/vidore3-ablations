"""In-memory ViDoRe dataset container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

GroundTruthBoxes = Dict[str, Dict[str, List[dict]]]


@dataclass
class VidoreDataset:
    query_ids: List[str]
    queries: List[str]
    corpus_ids: List[str]
    corpus_images: List[Any]
    corpus_texts: List[str]
    qrels: Dict[str, Dict[str, int]]
    query_languages: Dict[str, str]
    qrels_boxes: GroundTruthBoxes

    @property
    def num_queries(self) -> int:
        return len(self.query_ids)

    @property
    def num_corpus(self) -> int:
        return len(self.corpus_ids)

    def as_tuple(self):
        """Return the legacy 8-tuple used before the dataclass wrapper."""
        return (
            self.query_ids,
            self.queries,
            self.corpus_ids,
            self.corpus_images,
            self.corpus_texts,
            self.qrels,
            self.query_languages,
            self.qrels_boxes,
        )
