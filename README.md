# ViDoRe3 Technical Documents Retrieval Ablations

Ablation framework for comparing visual and text-based document retrieval techniques on the **technical documents** subset of [ViDoRe v3](https://huggingface.co/datasets/vidore/vidore_v3_industrial) (`vidore/vidore_v3_industrial`).

The industrial split contains ~5,244 pages from military aircraft technical manuals (fueling, mechanics, schematics) with 283 English queries (1,698 including translations). Each page includes a rendered image and OCR markdown.

## Ablations

| Name | Modality | Technique |
|------|----------|-----------|
| `random` | — | Random baseline |
| `bm25_ocr` | Text | BM25 on OCR markdown |
| `dense_text_ocr` | Text | Sentence-transformer bi-encoder on OCR |
| `clip_visual` | Visual | CLIP image–text similarity on page images |
| `clip_visual_half_res` | Visual | CLIP at 50% image resolution |
| `clip_text_on_ocr` | Text | CLIP text encoder on OCR (no images) |
| `hybrid_rrf` | Hybrid | Reciprocal rank fusion of text + visual |
| `hybrid_weighted` | Hybrid | Weighted score fusion (0.5/0.5) |
| `visual_no_text` | Visual | CLIP visual with OCR blanked |
| `colpali_late_interaction` | Visual | ColPali MaxSim late interaction (patch-level) |
| `jina_v5_omni_nano` | Visual | Jina Embeddings v5 Omni Nano retrieval embeddings on page images |

Metrics: NDCG@5/10, Recall@5/10, MAP (via `pytrec_eval`).

The project includes its own ViDoRe v3 dataset loader and evaluator (compatible with the [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark) pipeline API).

## Evaluation metrics

### Page retrieval (visual document search)

Standard IR metrics over page-level relevance judgments (`qrels`):

| Metric | Meaning |
|--------|---------|
| `ndcg_cut_k` | Ranking quality with graded relevance (1=critical, 2=full) |
| `recall_k` | Share of queries with any relevant page in top-k |
| `map` | Mean average precision across queries |

### Region grounding (manual evidence zones)

Human annotators mark bounding boxes on relevant pages. Following the [ViDoRe v3 protocol](https://arxiv.org/abs/2601.08620), each annotator's boxes are merged into a pixel zone and compared with predictions:

| Metric | Meaning |
|--------|---------|
| `grounding_iou_at_1` | Best-match **IoU** on the rank-1 retrieved page |
| `grounding_f1_at_1` | Best-match **F1 / Dice** on the rank-1 page |
| `grounding_precision_at_1` | Pixel precision of predicted zone on rank-1 |
| `grounding_recall_at_1` | Pixel recall of predicted zone on rank-1 |
| `grounding_iou_at_5` | Max best-match IoU among top-5 pages |
| `grounding_f1_at_5` | Max best-match F1 among top-5 pages |
| `box_precision_at_0.5` | Greedy box-level precision at IoU ≥ 0.5 |
| `box_recall_at_0.5` | Greedy box-level recall at IoU ≥ 0.5 |

**Best-match rule:** when multiple annotators labeled a page, the score uses the annotator with the highest F1 (ViDoRe accounts for subjective evidence selection).

### End-to-end success

| Metric | Meaning |
|--------|---------|
| `grounded_success_iou50_at_1` | Rank-1 page is relevant **and** zone IoU ≥ 0.5 |
| `grounded_success_f1_at_1` | Rank-1 page is relevant **and** zone F1 ≥ 0.5 |
| `localization_recall_at_k` | Any bbox-annotated relevant page appears in top-k |
| `retrieval_and_grounding_at_1` | Rank-1 page is relevant **and** grounding F1 > 0 |

Retrieval-only pipelines default to a **full-page** predicted zone via `BasePipeline.ground()`. Override `ground()` to evaluate region-specific models.

## Setup

```powershell
cd C:\Users\tsviv\Projects\vidore3-ablations
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

You need a Hugging Face account/token if the dataset requires authentication. Set `HF_TOKEN` or run `huggingface-cli login`.

## Quick smoke test

Run on a small subsample before the full benchmark (~5k pages):

```powershell
python -m vidore3_ablations.run_ablations --max-queries 20 --max-corpus 200 --ablations random bm25_ocr dense_text_ocr
```

Smoke test the Jina v5 Omni Nano benchmark separately; the first run downloads
`jinaai/jina-embeddings-v5-omni-nano-retrieval` and executes trusted remote
model code from Hugging Face:

```powershell
python -m vidore3_ablations.run_ablations --max-queries 5 --max-corpus 50 --ablations jina_v5_omni_nano
```

## Full ablation sweep

```powershell
python -m vidore3_ablations.run_ablations
```

Results are written to `results/<ablation>.json` plus `results/summary.json`.

## Analyze results

```powershell
python -m vidore3_ablations.analyze_results --results-dir results
```

## Extending

1. Add a pipeline class inheriting `BasePipeline` in `src/vidore3_ablations/pipelines/`.
2. Register it in `pipelines/registry.py`.
3. Add an entry under `ablations:` in `configs/ablations.yaml`.

ColPali requires a GPU with ~8GB+ VRAM for full-corpus indexing. Use `--max-corpus` for smoke tests on CPU.
Jina v5 Omni Nano requires `transformers>=5.0` and `torch>=2.5`; the default
ablation loads only the vision+text components via `modality: vision`.

## References

- [ViDoRe v3 paper](https://arxiv.org/abs/2601.08620)
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
- [Industrial dataset card](https://huggingface.co/datasets/vidore/vidore_v3_industrial)
