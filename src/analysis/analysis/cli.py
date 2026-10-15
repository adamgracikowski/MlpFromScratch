"""Command-line interface of the presentation layer."""

import argparse
import sys
from pathlib import Path

from analysis import __version__
from analysis.experiment import ArtifactsError, load_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the analysis CLI."""
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="Generate plots and statistics from MLP training artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    plot = commands.add_parser(
        "plot",
        help="plot one experiment",
        description=(
            "Read <experiment-dir>/artifacts/ and write plots and statistics to "
            "<experiment-dir>/plots/. Dataset paths in the config are relative to the "
            "current working directory, so run from the repository root."
        ),
    )
    plot.add_argument("experiment_dir", type=Path, metavar="<experiment-dir>")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the analysis CLI."""
    args = build_parser().parse_args(argv)
    try:
        experiment = load_experiment(args.experiment_dir)
    except ArtifactsError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    samples = {split: len(p.rows) for split, p in experiment.predictions.items()}
    print(f"task:     {experiment.task}")
    print(f"samples:  {samples}")
    print(f"stop:     {experiment.stop_reason}")
