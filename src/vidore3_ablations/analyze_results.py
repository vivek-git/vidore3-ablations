"""Summarize ablation JSON outputs into a comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

from vidore3_ablations.results_io import load_summary_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = load_summary_dataframe(args.results_dir)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output = args.output or args.results_dir / "summary.csv"
    df.to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
