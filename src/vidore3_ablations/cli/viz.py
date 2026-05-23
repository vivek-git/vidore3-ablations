"""Generate charts from ablation result JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from vidore3_ablations.results import generate_plots, load_summary_dataframe


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files (default: <results-dir>/figures)",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (args.results_dir / "figures")
    df = load_summary_dataframe(args.results_dir)
    paths = generate_plots(df, output_dir)

    print(f"Loaded {len(df)} ablations from {args.results_dir}")
    for path in paths:
        print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
