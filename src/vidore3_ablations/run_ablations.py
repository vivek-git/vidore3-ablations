"""Run configured ablation experiments on ViDoRe v3 technical documents."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm
from vidore3_ablations.data import load_vidore_dataset
from vidore3_ablations.evaluator import aggregate_results, evaluate_retrieval

from vidore3_ablations.pipelines.registry import build_pipeline


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "ablations.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def subsample_dataset(
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus_images,
    corpus_texts: List[str],
    qrels: Dict[str, Dict[str, int]],
    qrels_boxes: Dict[str, Dict[str, List[dict]]],
    query_languages: Dict[str, str],
    max_queries: Optional[int],
    max_corpus: Optional[int],
):
    if max_queries is not None:
        query_ids = query_ids[:max_queries]
        queries = queries[:max_queries]
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
        qrels_boxes = {qid: qrels_boxes[qid] for qid in query_ids if qid in qrels_boxes}
        query_languages = {qid: query_languages[qid] for qid in query_ids if qid in query_languages}

    if max_corpus is not None:
        keep_ids = set(corpus_ids[:max_corpus])
        corpus_ids = corpus_ids[:max_corpus]
        corpus_images = corpus_images[:max_corpus]
        corpus_texts = corpus_texts[:max_corpus]
        qrels = {
            qid: {cid: score for cid, score in rels.items() if cid in keep_ids}
            for qid, rels in qrels.items()
        }
        qrels_boxes = {
            qid: {cid: boxes for cid, boxes in rels.items() if cid in keep_ids}
            for qid, rels in qrels_boxes.items()
        }

    return query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, qrels_boxes, query_languages


def run_single_ablation(
    name: str,
    spec: Dict[str, Any],
    dataset_name: str,
    language: str,
    split: str,
    metrics: List[str],
    grounding_metrics: List[str],
    evaluate_grounding: bool,
    output_dir: Path,
    max_queries: Optional[int],
    max_corpus: Optional[int],
) -> Dict[str, Any]:
    print(f"\n=== Ablation: {name} ===")
    print(spec.get("description", ""))

    query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages, qrels_boxes = load_vidore_dataset(
        dataset_name=dataset_name,
        split=split,
        language=language,
    )
    query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, qrels_boxes, query_languages = subsample_dataset(
        query_ids,
        queries,
        corpus_ids,
        corpus_images,
        corpus_texts,
        qrels,
        qrels_boxes,
        query_languages,
        max_queries,
        max_corpus,
    )

    pipeline = build_pipeline(spec["pipeline"], **spec.get("args", {}))
    started = time.time()
    results = evaluate_retrieval(
        pipeline=pipeline,
        query_ids=query_ids,
        queries=queries,
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        qrels=qrels,
        qrels_boxes=qrels_boxes,
        dataset_name=dataset_name,
        metrics=metrics,
        evaluate_region_grounding=evaluate_grounding,
    )
    aggregated = aggregate_results(results, query_languages=query_languages)
    elapsed = time.time() - started

    payload = {
        "ablation": name,
        "description": spec.get("description"),
        "pipeline": spec["pipeline"],
        "pipeline_args": spec.get("args", {}),
        "dataset": dataset_name,
        "language": language,
        "split": split,
        "metrics_requested": metrics,
        "grounding_metrics_requested": grounding_metrics,
        "evaluate_grounding": evaluate_grounding,
        "num_queries": len(query_ids),
        "num_corpus": len(corpus_ids),
        "elapsed_seconds": elapsed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": aggregated,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved: {out_path}")

    overall = aggregated.get("overall", aggregated)
    ndcg = overall.get("ndcg_cut_10")
    recall = overall.get("recall_10")
    if ndcg is not None:
        print(f"NDCG@10: {ndcg:.4f}")
    if recall is not None:
        print(f"Recall@10: {recall:.4f}")
    for metric in grounding_metrics:
        value = overall.get(metric)
        if value is not None:
            print(f"{metric}: {value:.4f}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ablations", nargs="*", help="Run only these ablation names (default: all)")
    parser.add_argument("--max-queries", type=int, default=None, help="Override config subsample for queries")
    parser.add_argument("--max-corpus", type=int, default=None, help="Override config subsample for corpus")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ablation_names = args.ablations or list(config["ablations"].keys())
    output_dir = args.output_dir or Path(config.get("output_dir", "results"))
    max_queries = args.max_queries if args.max_queries is not None else config.get("max_queries")
    max_corpus = args.max_corpus if args.max_corpus is not None else config.get("max_corpus")

    summary_rows = []
    for name in tqdm(ablation_names, desc="Ablations"):
        if name not in config["ablations"]:
            raise ValueError(f"Unknown ablation '{name}' in config")
        payload = run_single_ablation(
            name=name,
            spec=config["ablations"][name],
            dataset_name=config["dataset"],
            language=config.get("language"),
            split=config.get("split", "test"),
            metrics=config.get("metrics", ["ndcg_cut_10"]),
            grounding_metrics=config.get("grounding_metrics", []),
            evaluate_grounding=config.get("evaluate_grounding", True),
            output_dir=output_dir,
            max_queries=max_queries,
            max_corpus=max_corpus,
        )
        overall = payload["results"].get("overall", payload["results"])
        all_metrics = list(config.get("metrics", [])) + list(config.get("grounding_metrics", []))
        summary_rows.append(
            {
                "ablation": name,
                "pipeline": payload["pipeline"],
                **{metric: overall.get(metric) for metric in all_metrics},
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
