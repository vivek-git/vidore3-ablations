"""Summarize ablation JSON outputs into a comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.results_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    else:
        rows = []
        for path in sorted(args.results_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            overall = payload["results"].get("overall", payload["results"])
            row = {
                "ablation": payload["ablation"],
                "pipeline": payload["pipeline"],
                "description": payload.get("description"),
            }
            for key, value in overall.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            rows.append(row)

    if not rows:
        raise SystemExit(f"No result files found in {args.results_dir}")

    df = pd.DataFrame(rows)
    metric_cols = [col for col in df.columns if col not in {"ablation", "pipeline", "description"}]
    sort_col = "ndcg_cut_10" if "ndcg_cut_10" in metric_cols else metric_cols[0]
    df = df.sort_values(sort_col, ascending=False)

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output = args.output or args.results_dir / "summary.csv"
    df.to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
