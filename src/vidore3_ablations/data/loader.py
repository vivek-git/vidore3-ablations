"""Load ViDoRe v3 datasets from HuggingFace."""

from __future__ import annotations

from typing import Dict, List

from datasets import load_dataset

from vidore3_ablations.data.dataset import GroundTruthBoxes, VidoreDataset

AVAILABLE_DATASETS = [
    "vidore/vidore_v3_hr",
    "vidore/vidore_v3_finance_en",
    "vidore/vidore_v3_industrial",
    "vidore/vidore_v3_pharmaceuticals",
    "vidore/vidore_v3_computer_science",
    "vidore/vidore_v3_energy",
    "vidore/vidore_v3_physics",
    "vidore/vidore_v3_finance_fr",
]

TECHNICAL_DOCUMENTS_DATASET = "vidore/vidore_v3_industrial"


def load_vidore_dataset(
    dataset_name: str,
    split: str = "test",
    language: str | None = None,
) -> VidoreDataset:
    """Load queries, corpus pages, qrels, and bounding boxes."""
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    queries_ds = load_dataset(dataset_name, data_dir="queries", split=split)
    corpus_ds = load_dataset(dataset_name, data_dir="corpus", split=split)
    qrels_ds = load_dataset(dataset_name, data_dir="qrels", split=split)

    if language:
        queries_ds = queries_ds.filter(lambda x: x["language"] == language)

    query_ids = [str(qid) for qid in queries_ds["query_id"]]
    queries = queries_ds["query"]
    query_languages = (
        dict(zip(query_ids, queries_ds["language"]))
        if "language" in queries_ds.column_names
        else {qid: "unknown" for qid in query_ids}
    )

    corpus_ids = [str(cid) for cid in corpus_ds["corpus_id"]]
    corpus_images = corpus_ds["image"]
    corpus_texts = corpus_ds["markdown"]

    query_id_set = set(query_ids)
    qrels: Dict[str, Dict[str, int]] = {}
    qrels_boxes: GroundTruthBoxes = {}
    for item in qrels_ds:
        query_id = str(item["query_id"])
        if query_id not in query_id_set:
            continue
        corpus_id = str(item["corpus_id"])
        qrels.setdefault(query_id, {})[corpus_id] = int(item["score"])
        boxes = item.get("bounding_boxes") or []
        if boxes:
            qrels_boxes.setdefault(query_id, {})[corpus_id] = list(boxes)

    if not queries:
        raise ValueError(f"No queries found in {dataset_name}")
    if not corpus_images:
        raise ValueError(f"No corpus images found in {dataset_name}")

    return VidoreDataset(
        query_ids=query_ids,
        queries=queries,
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        qrels=qrels,
        query_languages=query_languages,
        qrels_boxes=qrels_boxes,
    )
