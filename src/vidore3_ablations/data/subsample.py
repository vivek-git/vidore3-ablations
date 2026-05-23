"""Subsample ViDoRe datasets for smoke tests and exploration."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Set

from vidore3_ablations.data.dataset import VidoreDataset

CorpusStrategy = Literal["head", "qrels_first"]


def subsample_dataset(
    dataset: VidoreDataset,
    max_queries: int | None = None,
    max_corpus: int | None = None,
    corpus_strategy: CorpusStrategy = "qrels_first",
) -> VidoreDataset:
    """Return a smaller copy of the dataset, keeping qrel-linked pages when possible."""
    query_ids = dataset.query_ids
    queries = dataset.queries
    corpus_ids = list(dataset.corpus_ids)
    corpus_images = list(dataset.corpus_images)
    corpus_texts = list(dataset.corpus_texts)
    qrels = dict(dataset.qrels)
    qrels_boxes = dict(dataset.qrels_boxes)
    query_languages = dict(dataset.query_languages)

    if max_queries is not None:
        query_ids = query_ids[:max_queries]
        queries = queries[:max_queries]
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
        qrels_boxes = {qid: qrels_boxes[qid] for qid in query_ids if qid in qrels_boxes}
        query_languages = {qid: query_languages[qid] for qid in query_ids if qid in query_languages}

    if max_corpus is not None:
        keep = _select_corpus_ids(corpus_ids, qrels, max_corpus, corpus_strategy)
        corpus_ids, corpus_images, corpus_texts = _filter_corpus(
            corpus_ids, corpus_images, corpus_texts, keep
        )
        qrels = {qid: {cid: s for cid, s in rels.items() if cid in keep} for qid, rels in qrels.items()}
        qrels_boxes = {
            qid: {cid: b for cid, b in rels.items() if cid in keep} for qid, rels in qrels_boxes.items()
        }

    return replace(
        dataset,
        query_ids=query_ids,
        queries=queries,
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        qrels=qrels,
        qrels_boxes=qrels_boxes,
        query_languages=query_languages,
    )


def _select_corpus_ids(
    corpus_ids: list[str],
    qrels: dict[str, dict[str, int]],
    max_corpus: int,
    strategy: CorpusStrategy,
) -> Set[str]:
    if strategy == "head":
        return set(corpus_ids[:max_corpus])

    needed: Set[str] = set()
    for rels in qrels.values():
        needed.update(rels.keys())

    keep_order: list[str] = []
    for cid in corpus_ids:
        if cid in needed and cid not in keep_order:
            keep_order.append(cid)
    for cid in corpus_ids:
        if cid not in keep_order:
            keep_order.append(cid)
        if len(keep_order) >= max_corpus:
            break
    return set(keep_order[:max_corpus])


def _filter_corpus(corpus_ids, corpus_images, corpus_texts, keep: Set[str]):
    filtered_ids: list[str] = []
    filtered_images = []
    filtered_texts: list[str] = []
    for cid, image, text in zip(corpus_ids, corpus_images, corpus_texts):
        if cid in keep:
            filtered_ids.append(cid)
            filtered_images.append(image)
            filtered_texts.append(text)
    return filtered_ids, filtered_images, filtered_texts
