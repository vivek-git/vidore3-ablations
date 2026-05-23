"""Aggregate per-query metrics into overall and per-language summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional


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
