"""Generate charts from ablation result JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vidore3_ablations.results_io import (
    grounding_metrics,
    load_summary_dataframe,
    metric_columns,
    retrieval_metrics,
)


def _format_metric_label(metric: str) -> str:
    return metric.replace("_", " ").replace("at", "@").replace("cut", "")


def plot_grouped_bars(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    output_path: Path,
    ylabel: str = "Score",
) -> None:
    if not metrics:
        return

    labels = df["ablation"].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        values = df[metric].astype(float).tolist()
        bars = ax.bar(x + offset, values, width, label=_format_metric_label(metric))
        for bar, value in zip(bars, values):
            if value >= 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, min(1.05, max(df[metrics].max().max() * 1.15, 0.1)))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, metrics: list[str], title: str, output_path: Path) -> None:
    if not metrics:
        return

    matrix = df.set_index("ablation")[metrics].astype(float)
    normalized = matrix.copy()
    for col in normalized.columns:
        col_max = normalized[col].max()
        if col_max > 0:
            normalized[col] = normalized[col] / col_max

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.9), max(4, len(matrix) * 0.5)))
    im = ax.imshow(normalized.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([_format_metric_label(m) for m in metrics], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            raw = matrix.iloc[row, col]
            ax.text(col, row, f"{raw:.2f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="Normalized score (per metric column)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(df: pd.DataFrame, output_path: Path) -> None:
    if "elapsed_seconds" not in df.columns:
        return

    runtime = df.sort_values("elapsed_seconds", ascending=True)
    fig, ax = plt.subplots(figsize=(max(8, len(runtime) * 1.0), 4))
    bars = ax.barh(runtime["ablation"], runtime["elapsed_seconds"], color="#6b8cae")
    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_title("Ablation runtime")
    ax.grid(axis="x", alpha=0.3)
    for bar, seconds in zip(bars, runtime["elapsed_seconds"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {seconds:.1f}s", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    ret_metrics = retrieval_metrics(df)[:3] or metric_columns(df)[:3]
    grd_metrics = grounding_metrics(df)[:3]
    all_metrics = metric_columns(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("ViDoRe v3 Ablation Results", fontsize=14, fontweight="bold")

    # Retrieval panel
    ax = axes[0, 0]
    if ret_metrics:
        x = np.arange(len(df))
        width = 0.8 / len(ret_metrics)
        for idx, metric in enumerate(ret_metrics):
            offset = (idx - (len(ret_metrics) - 1) / 2) * width
            ax.bar(x + offset, df[metric], width, label=_format_metric_label(metric))
        ax.set_xticks(x)
        ax.set_xticklabels(df["ablation"], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    ax.set_title("Page retrieval")
    ax.grid(axis="y", alpha=0.3)

    # Grounding panel
    ax = axes[0, 1]
    if grd_metrics:
        x = np.arange(len(df))
        width = 0.8 / len(grd_metrics)
        for idx, metric in enumerate(grd_metrics):
            offset = (idx - (len(grd_metrics) - 1) / 2) * width
            ax.bar(x + offset, df[metric], width, label=_format_metric_label(metric))
        ax.set_xticks(x)
        ax.set_xticklabels(df["ablation"], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    ax.set_title("Region grounding & end-to-end")
    ax.grid(axis="y", alpha=0.3)

    # Heatmap panel
    ax = axes[1, 0]
    if all_metrics:
        matrix = df.set_index("ablation")[all_metrics].astype(float)
        norm = matrix.div(matrix.max().replace(0, 1), axis=1)
        im = ax.imshow(norm.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_metrics)))
        ax.set_xticklabels([_format_metric_label(m) for m in all_metrics], rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("All metrics (column-normalized)")

    # Runtime panel
    ax = axes[1, 1]
    if "elapsed_seconds" in df.columns:
        order = df.sort_values("elapsed_seconds", ascending=True)
        ax.barh(order["ablation"], order["elapsed_seconds"], color="#6b8cae")
        ax.set_xlabel("Seconds")
        ax.grid(axis="x", alpha=0.3)
    ax.set_title("Runtime")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    ret = retrieval_metrics(df)
    if ret:
        path = output_dir / "retrieval_comparison.png"
        plot_grouped_bars(df, ret, "Page retrieval metrics", path)
        written.append(path)

    grd = grounding_metrics(df)
    if grd:
        path = output_dir / "grounding_comparison.png"
        # Limit to 6 metrics for readability
        plot_grouped_bars(df, grd[:6], "Region grounding metrics", path)
        written.append(path)

    all_m = metric_columns(df)
    if all_m:
        path = output_dir / "metrics_heatmap.png"
        plot_heatmap(df, all_m, "Ablation metrics heatmap", path)
        written.append(path)

    if "elapsed_seconds" in df.columns:
        path = output_dir / "runtime_comparison.png"
        plot_runtime(df, path)
        written.append(path)

    path = output_dir / "dashboard.png"
    plot_dashboard(df, path)
    written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files (default: <results-dir>/figures)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "figures")
    df = load_summary_dataframe(args.results_dir)
    paths = generate_plots(df, output_dir)

    print(f"Loaded {len(df)} ablations from {args.results_dir}")
    for path in paths:
        print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
