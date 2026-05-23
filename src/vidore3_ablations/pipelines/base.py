"""Pipeline interface for retrieval and optional region grounding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from vidore3_ablations.metrics.grounding import Box, full_page_box


class BasePipeline(ABC):
    def index(
        self,
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[str],
        dataset_name: str | None = None,
    ) -> None:
        self.corpus_ids = corpus_ids
        self.corpus_images = corpus_images
        self.corpus_texts = corpus_texts

    @abstractmethod
    def search(
        self, query_ids: List[str], queries: List[str]
    ) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]]:
        raise NotImplementedError

    def ground(
        self,
        query_ids: List[str],
        queries: List[str],
        ranked_pages: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, List[Box]]]:
        """
        Optional region grounding on retrieved pages.

        Returns mapping query_id -> corpus_id -> list of predicted boxes (x1, y1, x2, y2).
        Default uses the full page as the predicted evidence zone.
        """
        corpus_id_to_image = {cid: img for cid, img in zip(self.corpus_ids, self.corpus_images)}
        predictions: Dict[str, Dict[str, List[Box]]] = {}
        for query_id, page_ids in ranked_pages.items():
            predictions[query_id] = {}
            for corpus_id in page_ids:
                image = corpus_id_to_image.get(corpus_id)
                if image is None:
                    continue
                predictions[query_id][corpus_id] = [full_page_box(image.width, image.height)]
        return predictions
