"""Run configured ablation experiments."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from vidore3_ablations.cli.config import DEFAULT_CONFIG, load_config
from vidore3_ablations.data import VidoreDataset, load_vidore_dataset, subsample_dataset
from vidore3_ablations.eval import aggregate_results, evaluate_retrieval
from vidore3_ablations.hardware import (
    VramProfile,
    apply_vram_profile_args,
    configure_pytorch_memory,
    detect_vram_profile,
    get_cuda_vram_gb,
)
from vidore3_ablations.pipelines.registry import build_pipeline


def resolve_vram_profile(args) -> VramProfile:
    if args.vram_profile == "auto":
        profile: VramProfile = detect_vram_profile()
    elif args.vram_profile == "default":
        profile = "default"
    else:
        profile = args.vram_profile  # type: ignore[assignment]

    if args.low_vram and profile == "default":
        profile = "low_12gb"
    return profile


def print_vram_profile(profile: VramProfile) -> None:
    if profile == "default":
        return
    vram = get_cuda_vram_gb()
    label = {"low_12gb": "~12 GB", "ultra_8gb": "~8 GB laptop"}.get(profile, profile)
    if vram is not None:
        print(f"VRAM profile '{label}' enabled ({vram:.1f} GB GPU detected).")
    else:
        print(f"VRAM profile '{label}' enabled.")


def run_single_ablation(
    name: str,
    spec: Dict[str, Any],
    dataset: VidoreDataset,
    dataset_name: str,
    language: str,
    split: str,
    metrics: List[str],
    grounding_metrics: List[str],
    evaluate_grounding: bool,
    output_dir: Path,
    vram_profile: VramProfile = "default",
) -> Dict[str, Any]:
    print(f"\n=== Ablation: {name} ===")
    print(spec.get("description", ""))

    pipeline_args = dict(spec.get("args", {}))
    if vram_profile != "default":
        pipeline_args = apply_vram_profile_args(spec["pipeline"], pipeline_args, vram_profile)

    pipeline = build_pipeline(spec["pipeline"], **pipeline_args)
    started = time.time()
    results = evaluate_retrieval(
        pipeline=pipeline,
        query_ids=dataset.query_ids,
        queries=dataset.queries,
        corpus_ids=dataset.corpus_ids,
        corpus_images=dataset.corpus_images,
        corpus_texts=dataset.corpus_texts,
        qrels=dataset.qrels,
        qrels_boxes=dataset.qrels_boxes,
        dataset_name=dataset_name,
        metrics=metrics,
        evaluate_region_grounding=evaluate_grounding,
    )
    aggregated = aggregate_results(results, query_languages=dataset.query_languages)
    elapsed = time.time() - started

    payload = {
        "ablation": name,
        "description": spec.get("description"),
        "pipeline": spec["pipeline"],
        "pipeline_args": pipeline_args,
        "dataset": dataset_name,
        "language": language,
        "split": split,
        "metrics_requested": metrics,
        "grounding_metrics_requested": grounding_metrics,
        "evaluate_grounding": evaluate_grounding,
        "vram_profile": vram_profile,
        "num_queries": dataset.num_queries,
        "num_corpus": dataset.num_corpus,
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ablations", nargs="*", help="Run only these ablation names (default: all)")
    parser.add_argument("--max-queries", type=int, default=None, help="Override config subsample for queries")
    parser.add_argument("--max-corpus", type=int, default=None, help="Override config subsample for corpus")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--vram-profile",
        choices=["auto", "default", "low_12gb", "ultra_8gb"],
        default="auto",
        help="GPU memory profile (auto detects 8 GB / 12 GB laptop GPUs)",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Alias for --vram-profile low_12gb",
    )
    args = parser.parse_args(argv)

    configure_pytorch_memory()
    vram_profile = resolve_vram_profile(args)
    print_vram_profile(vram_profile)

    config = load_config(args.config)
    config_profile = config.get("vram_profile")
    if config_profile and config_profile != "default" and vram_profile == "default":
        vram_profile = config_profile  # type: ignore[assignment]
        print(f"Using VRAM profile from config: {vram_profile}")

    ablation_names = args.ablations or list(config["ablations"].keys())
    output_dir = args.output_dir or Path(config.get("output_dir", "results"))
    max_queries = args.max_queries if args.max_queries is not None else config.get("max_queries")
    max_corpus = args.max_corpus if args.max_corpus is not None else config.get("max_corpus")

    print("Loading dataset...")
    dataset = load_vidore_dataset(
        dataset_name=config["dataset"],
        split=config.get("split", "test"),
        language=config.get("language"),
    )
    dataset = subsample_dataset(dataset, max_queries=max_queries, max_corpus=max_corpus)
    print(f"  {dataset.num_queries} queries, {dataset.num_corpus} corpus pages")

    summary_rows = []
    for name in tqdm(ablation_names, desc="Ablations"):
        if name not in config["ablations"]:
            raise ValueError(f"Unknown ablation '{name}' in config")
        payload = run_single_ablation(
            name=name,
            spec=config["ablations"][name],
            dataset=dataset,
            dataset_name=config["dataset"],
            language=config.get("language"),
            split=config.get("split", "test"),
            metrics=config.get("metrics", ["ndcg_cut_10"]),
            grounding_metrics=config.get("grounding_metrics", []),
            evaluate_grounding=config.get("evaluate_grounding", True),
            output_dir=output_dir,
            vram_profile=vram_profile,
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
