"""Command-line interface of the presentation layer."""

import argparse
import sys
from pathlib import Path

from analysis import __version__
from analysis.experiment import ArtifactsError, load_experiment
from analysis.plot_experiment import DEFAULT_DPI, DEFAULT_FORMATS, plot_experiment
from analysis.style import FORMATS


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
            "<experiment-dir>/plots/ (removed first). Dataset paths in the config are relative "
            "to the current working directory, so run from the repository root."
        ),
    )
    plot.add_argument("experiment_dir", type=Path, metavar="<experiment-dir>")
    plot.add_argument(
        "--format",
        dest="formats",
        type=_formats,
        default=list(DEFAULT_FORMATS),
        metavar="<formats>",
        help=f"comma-separated file formats out of {', '.join(FORMATS)} "
        f"(default: {','.join(DEFAULT_FORMATS)})",
    )
    plot.add_argument(
        "--dpi",
        type=_positive_int,
        default=DEFAULT_DPI,
        metavar="<dpi>",
        help=f"resolution of raster images (default: {DEFAULT_DPI})",
    )
    return parser


def _formats(value: str) -> list[str]:
    formats = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [part for part in formats if part not in FORMATS]
    if not formats or unknown:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list out of {', '.join(FORMATS)}, got {value!r}"
        )
    return list(dict.fromkeys(formats))


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def main(argv: list[str] | None = None) -> None:
    """Run the analysis CLI."""
    args = build_parser().parse_args(argv)
    try:
        experiment = load_experiment(args.experiment_dir)
    except ArtifactsError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    report = plot_experiment(experiment, formats=args.formats, dpi=args.dpi)
    for name, reason in report.skipped:
        print(f"skipped {name}: {reason}")
    print(f"plots:  {report.directory.as_posix()} ({len(report.files)} files)")
