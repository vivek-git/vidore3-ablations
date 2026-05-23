"""Page retrieval evaluation via pytrec_eval."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pytrec_eval

from vidore3_ablations.eval.grounding import evaluate_grounding
from vidore3_ablations.pipelines.base import BasePipeline


def _ranked_pages(run: Dict[str, Dict[str, float]], k: int) -> Dict[str, List[str]]:
    ranked: Dict[str, List[str]] = {}
    for query_id, scores in run.items():
        ranked[query_id] = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]
    return ranked


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
