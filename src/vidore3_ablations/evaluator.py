"""Evaluation utilities for retrieval and region grounding."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pytrec_eval

from vidore3_ablations.grounding import (
    Box,
    best_match_overlap,
    full_page_box,
    greedy_box_map,
    group_boxes_by_annotator,
    parse_boxes,
)
from vidore3_ablations.metric_definitions import (
    DEFAULT_F1_SUCCESS_THRESHOLD,
    DEFAULT_IOU_SUCCESS_THRESHOLD,
)
from vidore3_ablations.query_intent import QueryIntentGraph, QueryIntentGraphBuilder


class BasePipeline(ABC):
    def __init__(self) -> None:
        self.query_intent_graphs: Dict[str, QueryIntentGraph] = {}

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

    def set_query_intent_graphs(self, graphs: Dict[str, QueryIntentGraph]) -> None:
        """Attach query intent graphs for graph-aware pipelines."""

        self.query_intent_graphs = graphs

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


def _ranked_pages(run: Dict[str, Dict[str, float]], k: int) -> Dict[str, List[str]]:
    ranked: Dict[str, List[str]] = {}
    for query_id, scores in run.items():
        ranked[query_id] = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]
    return ranked


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


def evaluate_retrieval(
    pipeline: BasePipeline,
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus_images: List[Any],
    corpus_texts: List[str],
    qrels: Dict[str, Dict[str, int]],
    qrels_boxes: Optional[Mapping[str, Mapping[str, Sequence[dict]]]] = None,
    dataset_name: Optional[str] = None,
    metrics: List[str] | None = None,
    evaluate_region_grounding: bool = True,
    grounding_top_k: int = 10,
    track_time: bool = True,
) -> Dict[str, Dict[str, float]]:
    if metrics is None:
        metrics = ["ndcg_cut_10"]

    start_index = time.time()
    pipeline.index(
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        dataset_name=dataset_name,
    )
    indexing_time = time.time() - start_index
    if indexing_time < 1e-5:
        indexing_time = 0.0

    start_search = time.time()
    query_intent_graphs = QueryIntentGraphBuilder().build_many(query_ids, queries)
    pipeline.set_query_intent_graphs(query_intent_graphs)
    result = pipeline.search(query_ids=query_ids, queries=queries)
    search_time = time.time() - start_search

    if isinstance(result, tuple):
        run, infos = result
    else:
        run, infos = result, None

    for query_id in query_ids:
        run.setdefault(query_id, {})

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
    results = evaluator.evaluate(run)

    if evaluate_region_grounding and qrels_boxes:
        ranked = _ranked_pages(run, grounding_top_k)
        predictions = pipeline.ground(query_ids, queries, ranked)
        grounding_results = evaluate_grounding(
            run=run,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            corpus_images=corpus_images,
            qrels=qrels,
            qrels_boxes=qrels_boxes,
            predictions=predictions,
        )
        for query_id, grounding_metrics in grounding_results.items():
            results.setdefault(query_id, {}).update(grounding_metrics)

    if track_time:
        num_queries = len(query_ids)
        num_corpus = len(corpus_ids)
        results["_timing"] = {
            "total_retrieval_time_milliseconds": (indexing_time + search_time) * 1000,
            "indexing_time_milliseconds": indexing_time * 1000,
            "search_time_milliseconds": search_time * 1000,
            "num_queries": num_queries,
            "num_corpus": num_corpus,
        }
        if infos is not None:
            results["_infos"] = infos

    return results


def aggregate_results(
    results: Dict[str, Dict[str, float]],
    query_languages: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not results:
        return {}

    timing_info = results.pop("_timing", None)
    additional_infos = results.pop("_infos", None)

    if not results:
        final_result: Dict[str, Any] = {}
        if timing_info:
            final_result["timing"] = timing_info
        if additional_infos:
            final_result["infos"] = additional_infos
        return final_result

    metric_names = sorted({metric for query_results in results.values() for metric in query_results})

    def _average(metric: str, query_results_map: Dict[str, Dict[str, float]]) -> float:
        if not query_results_map:
            return 0.0
        return sum(row.get(metric, 0.0) for row in query_results_map.values()) / len(query_results_map)

    if query_languages is None:
        aggregated = {metric: _average(metric, results) for metric in metric_names}
        if timing_info:
            aggregated["timing"] = timing_info
        if additional_infos:
            aggregated["infos"] = additional_infos
        return aggregated

    results_by_language: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for query_id, query_results in results.items():
        lang = query_languages.get(query_id, "unknown")
        results_by_language[lang][query_id] = query_results

    overall = {metric: _average(metric, results) for metric in metric_names}
    by_language = {}
    for lang, lang_results in results_by_language.items():
        by_language[lang] = {
            **{metric: _average(metric, lang_results) for metric in metric_names},
            "num_queries": len(lang_results),
        }

    final_result = {"overall": overall, "by_language": by_language}
    if timing_info:
        final_result["timing"] = timing_info
    if additional_infos:
        final_result["infos"] = additional_infos
    return final_result
