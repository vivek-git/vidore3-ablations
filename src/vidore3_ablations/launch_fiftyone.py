"""Launch FiftyOne App to explore ViDoRe v3 documents and evidence regions."""

from __future__ import annotations

import argparse

from vidore3_ablations.fiftyone_integration import build_fiftyone_dataset, launch_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="vidore/vidore_v3_industrial",
        help="ViDoRe v3 HuggingFace dataset name",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="english")
    parser.add_argument(
        "--view",
        choices=["qrels", "corpus", "queries"],
        default="qrels",
        help="qrels=query-page pairs with boxes; corpus=all pages; queries=one row per query",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--port", type=int, default=None, help="FiftyOne App port (default: auto)")
    parser.add_argument("--dataset-name", default=None, help="FiftyOne dataset name")
    parser.add_argument("--persistent", action="store_true", help="Persist FiftyOne dataset on disk")
    parser.add_argument("--no-overwrite", action="store_true", help="Reuse existing FiftyOne dataset if present")
    args = parser.parse_args()

    try:
        import fiftyone  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'FiftyOne is not installed. Run: pip install -e ".[fiftyone]"'
        ) from exc

    print(f"Building FiftyOne dataset (view={args.view})...")
    dataset = build_fiftyone_dataset(
        dataset_name=args.dataset,
        split=args.split,
        language=args.language,
        view=args.view,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        fo_dataset_name=args.dataset_name,
        persistent=args.persistent,
        overwrite=not args.no_overwrite,
    )

    print(f"Loaded {len(dataset)} samples into FiftyOne dataset '{dataset.name}'")
    print("Launching FiftyOne App...")
    print("  - Open the 'ground_truth' field in the sidebar to toggle evidence regions")
    print("  - Filter by query_id, corpus_id, or relevance_score in the grid")
    print("  - Press Ctrl+C in this terminal to exit")

    session = launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
