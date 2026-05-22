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

## RTX 3060 / 12 GB VRAM

The defaults are tuned for consumer GPUs with ~12 GB VRAM:

| Pipeline | Memory strategy |
|----------|-----------------|
| **ColPali** | `batch_size=1`, fp16 weights, MaxSim scoring on **CPU** |
| **CLIP** | `batch_size=8`, fp16 autocast, embeddings stored on CPU |
| **Hybrid** | Text encoder on **CPU**, CLIP on GPU, models released between stages |
| **Dense text** | Runs on GPU by default; hybrid ablations pin it to CPU |

Low-VRAM mode activates automatically when ≤16 GB VRAM is detected, or force it with:

```powershell
python -m vidore3_ablations.run_ablations --low-vram
```

For the full industrial corpus (~5,244 pages), expect ~6 GB VRAM for ColPali weights plus headroom for single-page encoding. System RAM holds passage embeddings (~3–4 GB) while MaxSim runs on CPU.

Optional (Linux): `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set automatically at startup.

### RTX 4070 Laptop / 8 GB VRAM

Use the dedicated branch config and profile:

```powershell
git checkout rtx4070-8gb
python -m vidore3_ablations.run_ablations --config configs/ablations_rtx4070_8gb.yaml
# or on any config:
python -m vidore3_ablations.run_ablations --vram-profile ultra_8gb
```

8 GB settings vs 12 GB:

| Setting | 12 GB (`low_12gb`) | 8 GB laptop (`ultra_8gb`) |
|---------|-------------------|---------------------------|
| ColPali batch | 1 | 1 |
| ColPali MaxSim | CPU, batch 16 | CPU, batch 8 |
| ColPali GPU cap | full GPU | 7 GiB + CPU offload |
| CLIP batch | 8 | 4 @ 75% resolution |
| Text encoders | CPU in hybrid | CPU everywhere |

Auto-detection picks `ultra_8gb` when ≤8.5 GB VRAM is reported.

## Quick smoke test

Run on a small subsample before the full benchmark (~5k pages):

```powershell
python -m vidore3_ablations.run_ablations --max-queries 20 --max-corpus 200 --ablations random bm25_ocr dense_text_ocr
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

## Visualize results

Generate comparison charts (PNG) from `results/summary.json` or individual ablation JSON files:

```powershell
python -m vidore3_ablations.visualize_results --results-dir results
# or: vidore3-viz --results-dir results
```

Outputs to `results/figures/`:
- `dashboard.png` — 4-panel overview (retrieval, grounding, heatmap, runtime)
- `retrieval_comparison.png` — NDCG / recall / MAP bars
- `grounding_comparison.png` — IoU, F1, localization bars
- `metrics_heatmap.png` — all metrics with column-normalized colors
- `runtime_comparison.png` — elapsed seconds per ablation

```powershell
python -m vidore3_ablations.visualize_results --results-dir results --output-dir reports/figures
```

## Explore with FiftyOne

Interactively browse pages, queries, and human evidence bounding boxes:

```powershell
pip install -e ".[fiftyone]"
python -m vidore3_ablations.launch_fiftyone --view qrels --max-queries 50 --max-corpus 200
# or: vidore3-explore --view corpus
```

**Views**

| View | Samples | Best for |
|------|---------|----------|
| `qrels` | One per (query, relevant page) | Navigating evidence regions per query |
| `corpus` | One per document page | Browsing all pages and aggregated regions |
| `queries` | One per query | Query-centric exploration with preview page |

In the FiftyOne App:
1. Enable the **`ground_truth`** label field in the sidebar to show evidence boxes
2. Filter samples by `query_id`, `corpus_id`, or `relevance_score`
3. Click samples to pan/zoom regions; inspect `query_text` and `markdown_preview` metadata

Images are cached under `~/.cache/vidore3-ablations/fiftyone/`.


1. Add a pipeline class inheriting `BasePipeline` in `src/vidore3_ablations/pipelines/`.
2. Register it in `pipelines/registry.py`.
3. Add an entry under `ablations:` in `configs/ablations.yaml`.

ColPali requires a GPU with ~8GB+ VRAM for full-corpus indexing on a 12 GB card with the default settings above. Use `--max-corpus` for quick iteration.

## References

- [ViDoRe v3 paper](https://arxiv.org/abs/2601.08620)
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
- [Industrial dataset card](https://huggingface.co/datasets/vidore/vidore_v3_industrial)
