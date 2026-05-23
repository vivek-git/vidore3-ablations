"""Backward-compatible entry point. Prefer: vidore3 run or vidore3_ablations.cli.run."""

from vidore3_ablations.cli.run import main

if __name__ == "__main__":
    main()
