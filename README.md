# ViDoRe3 Technical Documents Retrieval Ablations

Ablation framework for comparing visual and text-based document retrieval on the **technical documents** subset of [ViDoRe v3](https://huggingface.co/datasets/vidore/vidore_v3_industrial) (`vidore/vidore_v3_industrial`).

The industrial split contains ~5,244 pages from military aircraft technical manuals with 283 English queries. Each page includes a rendered image and OCR markdown.

## Quick start

```powershell
cd C:\Users\tsviv\Projects\vidore3-ablations
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Smoke test (10 queries, 100 pages, 4 ablations)
vidore3 run --max-queries 10 --max-corpus 100 --ablations random bm25_ocr dense_text_ocr clip_visual

# Full sweep
vidore3 run

# Results
vidore3 analyze --results-dir results
vidore3 viz --results-dir results
```

Set `HF_TOKEN` or run `huggingface-cli login` if the dataset requires authentication.

## Commands

| Command | What it does |
|---------|--------------|
| `vidore3 run` | Run ablation experiments from YAML config |
| `vidore3 analyze` | Print comparison table + write `summary.csv` |
| `vidore3 viz` | Generate PNG charts in `results/figures/` |
| `vidore3 explore` | Launch FiftyOne to browse pages and evidence boxes |

All subcommands accept `--help`. Legacy entry points (`vidore3-ablate`, `python -m vidore3_ablations.run_ablations`, etc.) still work.

### Common options

```powershell
# 8 GB laptop GPU profile
vidore3 run --config configs/ablations_rtx4070_8gb.yaml --vram-profile ultra_8gb

# Subsample for quick iteration
vidore3 run --max-queries 20 --max-corpus 200 --ablations random bm25_ocr

# Interactive dataset explorer (requires pip install -e ".[fiftyone]")
vidore3 explore --view qrels --max-queries 50 --max-corpus 200
```

## Project layout

```
src/vidore3_ablations/
├── cli/           # run, analyze, viz, explore commands
├── data/          # HuggingFace loader + subsampling
├── eval/          # retrieval + grounding evaluation
├── explore/       # FiftyOne dataset builder
├── hardware/      # GPU device helpers + VRAM profiles
├── metrics/       # metric definitions + box overlap math
├── pipelines/     # retrieval methods (BM25, CLIP, ColPali, …)
└── results/       # load JSON results + plot charts
configs/
├── ablations.yaml              # full 10-ablation sweep
└── ablations_rtx4070_8gb.yaml  # 6-ablation subset for 8 GB GPUs
```

## Ablations

| Config name | Pipeline | Modality |
|-------------|----------|----------|
| `random` | `random` | Baseline |
| `bm25_ocr` | `bm25_text` | Text (BM25 on OCR) |
| `dense_text_ocr` | `dense_text` | Text (bi-encoder) |
| `clip_visual` | `clip_visual` | Visual (CLIP) |
| `clip_visual_half_res` | `clip_visual` | Visual (50% resolution) |
| `clip_text_on_ocr` | `clip_text_on_ocr` | Text (CLIP text encoder) |
| `hybrid_rrf` | `hybrid_rrf` | Hybrid (RRF fusion) |
| `hybrid_weighted` | `hybrid_weighted` | Hybrid (score fusion) |
| `visual_no_text` | `clip_visual` | Visual (OCR blanked) |
| `colpali_late_interaction` | `colpali_late_interaction` | Visual (ColPali MaxSim) |

Config names describe the experiment; pipeline names are the implementation keys in `pipelines/registry.py`.

## Evaluation metrics

### Page retrieval

| Metric | Meaning |
|--------|---------|
| `ndcg_cut_k` | Ranking quality with graded relevance |
| `recall_k` | Share of queries with a relevant page in top-k |
| `map` | Mean average precision |

### Region grounding

Human annotators mark bounding boxes on relevant pages. Scores use the best match across annotators (ViDoRe v3 protocol):

| Metric | Meaning |
|--------|---------|
| `grounding_iou_at_1` / `grounding_f1_at_1` | Zone overlap on rank-1 page |
| `grounding_iou_at_5` / `grounding_f1_at_5` | Best overlap among top-5 pages |
| `box_precision_at_0.5` / `box_recall_at_0.5` | Greedy box match at IoU ≥ 0.5 |
| `grounded_success_iou50_at_1` | Rank-1 relevant **and** IoU ≥ 0.5 |
| `localization_recall_at_k` | Annotated page appears in top-k |
| `retrieval_and_grounding_at_1` | Rank-1 relevant **and** grounding F1 > 0 |

Pipelines default to a full-page predicted zone via `BasePipeline.ground()`. Override `ground()` for region-specific models.

## GPU memory

VRAM profiles apply automatically based on detected GPU memory, or set explicitly:

| Profile | When | Key settings |
|---------|------|--------------|
| `low_12gb` | ≤16 GB | ColPali MaxSim on CPU, text encoders on CPU in hybrids |
| `ultra_8gb` | ≤8.5 GB | Smaller batches, 7 GiB GPU cap for ColPali |

```powershell
vidore3 run --vram-profile ultra_8gb
vidore3 run --config configs/ablations_rtx4070_8gb.yaml
```

## FiftyOne explorer

```powershell
pip install -e ".[fiftyone]"
vidore3 explore --view qrels    # one sample per (query, relevant page)
vidore3 explore --view corpus   # one sample per document page
vidore3 explore --view queries  # one sample per query
```

Enable the **`ground_truth`** field in the sidebar to show evidence bounding boxes. Images cache under `~/.cache/vidore3-ablations/fiftyone/`.

## Extending

1. Add a pipeline class in `src/vidore3_ablations/pipelines/` inheriting `BasePipeline`.
2. Register it in `pipelines/registry.py`.
3. Add an entry under `ablations:` in `configs/ablations.yaml`.

Config validation runs at startup and fails fast on unknown pipeline names.

## References

- [ViDoRe v3 paper](https://arxiv.org/abs/2601.08620)
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
- [Industrial dataset card](https://huggingface.co/datasets/vidore/vidore_v3_industrial)
