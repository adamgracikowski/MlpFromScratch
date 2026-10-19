"""Command-line interface of the presentation layer."""

import argparse
import sys
from pathlib import Path

from analysis import __version__
from analysis.aggregate_study import aggregate_study
from analysis.datasets import DatasetError, prepare_mnist
from analysis.experiment import ArtifactsError, load_experiment
from analysis.plot_experiment import DEFAULT_DPI, DEFAULT_FORMATS, plot_experiment
from analysis.study import StudyError, load_study
from analysis.style import FORMATS

MNIST_DIR = "data/datasets/mnist"


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
    _add_figure_options(plot)

    aggregate = commands.add_parser(
        "aggregate",
        help="aggregate the runs of a study over seeds",
        description=(
            "Read the artifacts of every run listed in <study-dir>/runs/index.csv and write "
            "tables aggregated over seeds, comparison plots and statistics to "
            "<study-dir>/results/ (removed first). Runs without artifacts are reported as "
            "missing."
        ),
    )
    aggregate.add_argument("study_dir", type=Path, metavar="<study-dir>")
    _add_figure_options(aggregate)
    aggregate.add_argument(
        "--plot-runs",
        action="store_true",
        help="also plot the representative run of every variant (closest to the median of "
        "the test metric) into its plots/; needs the datasets, so run from the repository root",
    )

    mnist = commands.add_parser(
        "mnist",
        help="download MNIST and convert it to CSV",
        description=(
            "Put the four IDX files of MNIST into <output>/raw/ (downloaded from the mirror, or "
            "taken from --source), verify their SHA-256 and write the CSV files read by the "
            "compute layer. Verified raw files and existing CSV files are kept."
        ),
    )
    mnist.add_argument(
        "--source",
        type=Path,
        metavar="<path>",
        help="directory or ZIP archive with the IDX files (e.g. the archive downloaded from "
        "Kaggle) instead of downloading them",
    )
    mnist.add_argument(
        "--output",
        type=Path,
        default=Path(MNIST_DIR),
        metavar="<dir>",
        help=f"directory of the dataset (default: {MNIST_DIR})",
    )
    mnist.add_argument(
        "--force", action="store_true", help="write the CSV files even if they exist"
    )
    return parser


def _add_figure_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        dest="formats",
        type=_formats,
        default=list(DEFAULT_FORMATS),
        metavar="<formats>",
        help=f"comma-separated file formats out of {', '.join(FORMATS)} "
        f"(default: {','.join(DEFAULT_FORMATS)})",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=DEFAULT_DPI,
        metavar="<dpi>",
        help=f"resolution of raster images (default: {DEFAULT_DPI})",
    )


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
        if args.command == "plot":
            _plot(args)
        elif args.command == "aggregate":
            _aggregate(args)
        else:
            _mnist(args)
    except (ArtifactsError, StudyError, DatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _plot(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.experiment_dir)
    report = plot_experiment(experiment, formats=args.formats, dpi=args.dpi)
    for name, reason in report.skipped:
        print(f"skipped {name}: {reason}")
    print(f"plots:  {report.directory.as_posix()} ({len(report.files)} files)")


def _aggregate(args: argparse.Namespace) -> None:
    study = load_study(args.study_dir)
    report = aggregate_study(study, formats=args.formats, dpi=args.dpi, plot_runs=args.plot_runs)
    for name, reason in report.skipped:
        print(f"skipped {name}: {reason}")
    if report.missing:
        print(f"warning: {len(report.missing)} run(s) without artifacts:", file=sys.stderr)
        for run in report.missing:
            print(f"  {run}", file=sys.stderr)
    trained = study.index.height - len(report.missing)
    print(f"runs:    {trained} of {study.index.height} trained")
    print(f"results: {report.directory.as_posix()} ({len(report.files)} files)")
    for variant, run in report.plotted_runs.items():
        print(f"plotted: {variant}: {run.as_posix()}/plots")


def _mnist(args: argparse.Namespace) -> None:
    report = prepare_mnist(args.output, source=args.source, force=args.force)
    for path, description in report.files:
        print(f"{path.as_posix()}: {description}")
