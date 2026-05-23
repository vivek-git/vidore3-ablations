# Smoke test: small subsample, lightweight ablations only
vidore3 run `
  --max-queries 10 `
  --max-corpus 100 `
  --ablations random bm25_ocr dense_text_ocr clip_visual

vidore3 analyze --results-dir results
