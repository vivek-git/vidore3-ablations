"""Build FiftyOne datasets from ViDoRe v3 for interactive exploration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, Set

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from vidore3_ablations.data import AVAILABLE_DATASETS, load_vidore_dataset, subsample_dataset
from vidore3_ablations.explore.boxes import boxes_to_fiftyone_detections

ViewMode = Literal["qrels", "corpus", "queries"]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vidore3-ablations" / "fiftyone"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def _image_cache_path(cache_dir: Path, corpus_id: str) -> Path:
    return cache_dir / "images" / f"{_safe_slug(corpus_id)}.png"


def _write_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image.convert("RGB").save(path)


def build_fiftyone_dataset(
    dataset_name: str = "vidore/vidore_v3_industrial",
    split: str = "test",
    language: str | None = "english",
    view: ViewMode = "qrels",
    max_queries: int | None = None,
    max_corpus: int | None = None,
    cache_dir: Path | None = None,
    fo_dataset_name: str | None = None,
    persistent: bool = False,
    overwrite: bool = True,
):
    """
    Create a FiftyOne dataset for interactive browsing of ViDoRe v3 pages and regions.

    Views:
    - qrels: one sample per (query, relevant page) with ground-truth evidence boxes
    - corpus: one sample per page with all evidence boxes from any query
    - queries: one sample per query (first relevant page thumbnail + metadata)
    """
    import fiftyone as fo

    cache_dir = cache_dir or (DEFAULT_CACHE_DIR / _safe_slug(dataset_name))
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_vidore_dataset(dataset_name=dataset_name, split=split, language=language)
    dataset = subsample_dataset(dataset, max_queries=max_queries, max_corpus=max_corpus)

    query_ids = dataset.query_ids
    queries = dataset.queries
    corpus_ids = dataset.corpus_ids
    corpus_images = dataset.corpus_images
    corpus_texts = dataset.corpus_texts
    qrels = dataset.qrels
    query_languages = dataset.query_languages
    qrels_boxes = dataset.qrels_boxes

    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    query_id_to_text = dict(zip(query_ids, queries))

    qrels_ds = load_dataset(dataset_name, data_dir="qrels", split=split)
    qrel_meta: Dict[tuple[str, str], dict] = {}
    query_set = set(query_ids)
    corpus_set = set(corpus_ids)
    for item in qrels_ds:
        qid, cid = str(item["query_id"]), str(item["corpus_id"])
        if qid in query_set and cid in corpus_set:
            qrel_meta[(qid, cid)] = item

    slug = _safe_slug(dataset_name.split("/")[-1])
    name = fo_dataset_name or f"vidore_{slug}_{view}"
    if fo.dataset_exists(name):
        if overwrite:
            fo.delete_dataset(name)
        else:
            return fo.load_dataset(name)

    fo_dataset = fo.Dataset(name, persistent=persistent)
    fo_dataset.description = f"ViDoRe v3 explorer ({view}) for {dataset_name}"

    if view == "corpus":
        _build_corpus_view(
            fo_dataset,
            corpus_ids,
            corpus_images,
            corpus_texts,
            qrels_boxes,
            query_id_to_text,
            cache_dir,
        )
    elif view == "queries":
        _build_queries_view(
            fo_dataset,
            query_ids,
            queries,
            query_languages,
            qrels,
            qrels_boxes,
            corpus_ids,
            corpus_images,
            corpus_texts,
            cache_dir,
        )
    else:
        _build_qrels_view(
            fo_dataset,
            query_ids,
            qrels,
            qrels_boxes,
            qrel_meta,
            query_id_to_text,
            query_languages,
            corpus_ids,
            corpus_images,
            corpus_texts,
            corpus_id_to_idx,
            cache_dir,
        )

    fo_dataset.compute_metadata()
    fo_dataset.save()
    return fo_dataset


def _build_qrels_view(
    dataset,
    query_ids,
    qrels,
    qrels_boxes,
    qrel_meta,
    query_id_to_text,
    query_languages,
    corpus_ids,
    corpus_images,
    corpus_texts,
    corpus_id_to_idx,
    cache_dir,
) -> None:
    import fiftyone as fo

    samples = []
    for query_id in tqdm(query_ids, desc="Building qrels view"):
        for corpus_id, score in qrels.get(query_id, {}).items():
            if corpus_id not in corpus_id_to_idx:
                continue
            idx = corpus_id_to_idx[corpus_id]
            image = corpus_images[idx]
            image_path = _image_cache_path(cache_dir, corpus_id)
            _write_image(image, image_path)

            meta = qrel_meta.get((query_id, corpus_id), {})
            raw_boxes = qrels_boxes.get(query_id, {}).get(corpus_id, [])
            gt = boxes_to_fiftyone_detections(raw_boxes, image.width, image.height)

            markdown = corpus_texts[idx] or ""
            sample = fo.Sample(
                filepath=str(image_path),
                query_id=query_id,
                query_text=query_id_to_text.get(query_id, ""),
                query_language=query_languages.get(query_id, "unknown"),
                corpus_id=corpus_id,
                relevance_score=int(score),
                content_type=str(meta.get("content_type", "")),
                has_bounding_boxes=bool(raw_boxes),
                num_bounding_boxes=len(raw_boxes),
                markdown_preview=markdown[:500],
                ground_truth=gt,
            )
            samples.append(sample)

    dataset.add_samples(samples)


def _build_corpus_view(
    dataset,
    corpus_ids,
    corpus_images,
    corpus_texts,
    qrels_boxes,
    query_id_to_text,
    cache_dir,
) -> None:
    import fiftyone as fo

    page_boxes: Dict[str, List] = {cid: [] for cid in corpus_ids}
    page_queries: Dict[str, Set[str]] = {cid: set() for cid in corpus_ids}
    for query_id, pages in qrels_boxes.items():
        for corpus_id, boxes in pages.items():
            if corpus_id not in page_boxes:
                continue
            page_boxes[corpus_id].extend(boxes)
            page_queries[corpus_id].add(query_id)

    samples = []
    for corpus_id, image in tqdm(zip(corpus_ids, corpus_images), total=len(corpus_ids), desc="Building corpus view"):
        image_path = _image_cache_path(cache_dir, corpus_id)
        _write_image(image, image_path)
        raw_boxes = page_boxes.get(corpus_id, [])
        gt = boxes_to_fiftyone_detections(raw_boxes, image.width, image.height, label_prefix="region")

        idx = corpus_ids.index(corpus_id)
        markdown = corpus_texts[idx] or ""
        linked_queries = sorted(page_queries.get(corpus_id, []))

        sample = fo.Sample(
            filepath=str(image_path),
            corpus_id=corpus_id,
            num_queries=len(linked_queries),
            linked_query_ids=linked_queries[:20],
            linked_query_texts=[query_id_to_text.get(q, "")[:120] for q in linked_queries[:5]],
            num_bounding_boxes=len(raw_boxes),
            markdown_preview=markdown[:500],
            ground_truth=gt,
        )
        samples.append(sample)

    dataset.add_samples(samples)


def _build_queries_view(
    dataset,
    query_ids,
    queries,
    query_languages,
    qrels,
    qrels_boxes,
    corpus_ids,
    corpus_images,
    corpus_texts,
    cache_dir,
) -> None:
    import fiftyone as fo

    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    samples = []

    for query_id, query_text in tqdm(zip(query_ids, queries), total=len(query_ids), desc="Building queries view"):
        rel_pages = qrels.get(query_id, {})
        if not rel_pages:
            continue

        corpus_id = max(rel_pages.items(), key=lambda item: item[1])[0]
        if corpus_id not in corpus_id_to_idx:
            continue
        idx = corpus_id_to_idx[corpus_id]
        image = corpus_images[idx]
        image_path = _image_cache_path(cache_dir, f"query_{query_id}_page_{corpus_id}")
        _write_image(image, image_path)

        raw_boxes = qrels_boxes.get(query_id, {}).get(corpus_id, [])
        gt = boxes_to_fiftyone_detections(raw_boxes, image.width, image.height)

        sample = fo.Sample(
            filepath=str(image_path),
            query_id=query_id,
            query_text=query_text,
            query_language=query_languages.get(query_id, "unknown"),
            num_relevant_pages=len(rel_pages),
            relevant_corpus_ids=list(rel_pages.keys()),
            preview_corpus_id=corpus_id,
            preview_relevance_score=int(rel_pages[corpus_id]),
            num_bounding_boxes=len(raw_boxes),
            ground_truth=gt,
        )
        samples.append(sample)

    dataset.add_samples(samples)


def launch_app(dataset, port: int | None = None):
    import fiftyone as fo

    session = fo.launch_app(dataset, port=port)
    return session
