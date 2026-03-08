"""Compatibility wrapper for the canonical scripts/cli.py entrypoint."""

from __future__ import annotations

import argparse

from scripts.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backbone tracks clustering pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/backbone.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
