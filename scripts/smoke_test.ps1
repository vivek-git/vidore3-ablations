# Smoke test: small subsample, lightweight ablations only
python -m vidore3_ablations.run_ablations `
  --max-queries 10 `
  --max-corpus 100 `
  --ablations random bm25_ocr dense_text_ocr clip_visual

python -m vidore3_ablations.analyze_results --results-dir results
