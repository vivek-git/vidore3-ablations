"""Load ablation result files into pandas DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from vidore3_ablations.metrics.definitions import END_TO_END_METRICS, GROUNDING_METRICS, RETRIEVAL_METRICS

META_COLUMNS = {"ablation", "pipeline", "description", "elapsed_seconds"}


def load_summary_dataframe(results_dir: Path) -> pd.DataFrame:
    """Load ablation metrics from summary.json or individual result JSON files."""
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    else:
        rows = []
        for path in sorted(results_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            overall = payload["results"].get("overall", payload["results"])
            row = {
                "ablation": payload["ablation"],
                "pipeline": payload["pipeline"],
                "description": payload.get("description"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
            }
            for key, value in overall.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    df = pd.DataFrame(rows)
    sort_col = "ndcg_cut_10" if "ndcg_cut_10" in df.columns else df.columns[0]
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    return df.reset_index(drop=True)


def metric_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])]


def metrics_in_df(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [metric for metric in candidates if metric in df.columns]


def retrieval_metrics(df: pd.DataFrame) -> List[str]:
    return metrics_in_df(df, RETRIEVAL_METRICS)


def grounding_metrics(df: pd.DataFrame) -> List[str]:
    return metrics_in_df(df, GROUNDING_METRICS + END_TO_END_METRICS)
