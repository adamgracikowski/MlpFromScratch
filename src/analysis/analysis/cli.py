"""Command-line interface of the presentation layer."""

import argparse

from analysis import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the analysis CLI."""
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="Generate plots and statistics from MLP training artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the analysis CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
