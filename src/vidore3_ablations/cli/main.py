"""Unified command-line interface for ViDoRe v3 ablations."""

from __future__ import annotations

import argparse
import sys


COMMANDS = {
    "run": ("vidore3_ablations.cli.run", "Run ablation experiments"),
    "analyze": ("vidore3_ablations.cli.analyze", "Print results table and write summary.csv"),
    "viz": ("vidore3_ablations.cli.viz", "Generate comparison charts"),
    "explore": ("vidore3_ablations.cli.explore", "Launch FiftyOne dataset explorer"),
}


def _print_top_level_help() -> None:
    print("usage: vidore3 <command> [options]")
    print("\nViDoRe v3 technical documents retrieval ablations\n")
    print("commands:")
    for name, (_, help_text) in COMMANDS.items():
        print(f"  {name:<10} {help_text}")
    print("\nRun 'vidore3 <command> --help' for command-specific options.")


def main(argv=None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_level_help()
        return

    command = argv[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        _print_top_level_help()
        raise SystemExit(2)

    module_path, _ = COMMANDS[command]
    import importlib

    module = importlib.import_module(module_path)
    module.main(argv[1:])


if __name__ == "__main__":
    main()
