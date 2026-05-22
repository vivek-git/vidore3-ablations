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

## References

- [ViDoRe v3 paper](https://arxiv.org/abs/2601.08620)
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
- [Industrial dataset card](https://huggingface.co/datasets/vidore/vidore_v3_industrial)
