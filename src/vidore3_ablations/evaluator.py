"""Evaluation utilities using pytrec_eval."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pytrec_eval


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


def evaluate_retrieval(
    pipeline: BasePipeline,
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus_images: List[Any],
    corpus_texts: List[str],
    qrels: Dict[str, Dict[str, int]],
    dataset_name: Optional[str] = None,
    metrics: List[str] | None = None,
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

    metric_names = list(next(iter(results.values())).keys())

    if query_languages is None:
        aggregated = {metric: sum(results[qid][metric] for qid in results) / len(results) for metric in metric_names}
        if timing_info:
            aggregated["timing"] = timing_info
        if additional_infos:
            aggregated["infos"] = additional_infos
        return aggregated

    results_by_language: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for query_id, query_results in results.items():
        lang = query_languages.get(query_id, "unknown")
        results_by_language[lang][query_id] = query_results

    overall = {metric: sum(results[qid][metric] for qid in results) / len(results) for metric in metric_names}
    by_language = {}
    for lang, lang_results in results_by_language.items():
        by_language[lang] = {
            **{metric: sum(lang_results[qid][metric] for qid in lang_results) / len(lang_results) for metric in metric_names},
            "num_queries": len(lang_results),
        }

    final_result = {"overall": overall, "by_language": by_language}
    if timing_info:
        final_result["timing"] = timing_info
    if additional_infos:
        final_result["infos"] = additional_infos
    return final_result
