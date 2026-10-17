"""Results of a study aggregated over seeds, written to `<study>/results/`."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analysis.experiment import load_experiment
from analysis.plot_experiment import DEFAULT_DPI, DEFAULT_FORMATS, plot_experiment
from analysis.plots.study_curves import plot_study_curves, plot_study_train_loss, training_losses
from analysis.plots.study_gallery import gallery_skip_reason, plot_study_gallery
from analysis.plots.study_heatmap import heatmap_skip_reason, plot_study_heatmap
from analysis.plots.study_seeds import plot_study_seeds
from analysis.study import (
    Study,
    StudyError,
    StudyResults,
    load_results,
    representative_runs,
    variants_table,
)
from analysis.study.range_errors import INSIDE, OUTSIDE
from analysis.study_summary import write_study_summary
from analysis.style import STYLE, save_figure

RESULTS_DIR = "results"


@dataclass(frozen=True)
class StudyPlot:
    """One comparison plot; a plot that does not apply (another task) is left out silently,
    a plot with a skip reason is left out and reported."""

    name: str
    draw: Callable[[StudyResults], Figure]
    skip_reason: Callable[[StudyResults], str | None] = lambda _: None
    applies: Callable[[StudyResults], bool] = lambda _: True  # false: left out silently


def _train_loss_reason(results: StudyResults) -> str | None:
    losses = training_losses(results)
    if len(losses) > 1:
        return f"the variants use different losses ({', '.join(losses)})"
    return None


def _regression(results: StudyResults) -> bool:
    return not results.study.is_classification


STUDY_PLOTS = [
    StudyPlot("seeds", plot_study_seeds),
    StudyPlot("seeds_inside", partial(plot_study_seeds, column=INSIDE), applies=_regression),
    StudyPlot("seeds_outside", partial(plot_study_seeds, column=OUTSIDE), applies=_regression),
    StudyPlot("curves", plot_study_curves),
    StudyPlot("curves_train_loss", plot_study_train_loss, _train_loss_reason),
    StudyPlot("heatmap", plot_study_heatmap, heatmap_skip_reason),
    StudyPlot(
        "heatmap_inside",
        partial(plot_study_heatmap, column=INSIDE),
        heatmap_skip_reason,
        _regression,
    ),
    StudyPlot(
        "heatmap_outside",
        partial(plot_study_heatmap, column=OUTSIDE),
        heatmap_skip_reason,
        _regression,
    ),
]


@dataclass(frozen=True)
class AggregateReport:
    """Written files, skipped plots, runs without artifacts and the plotted runs."""

    directory: Path
    files: list[Path]
    skipped: list[tuple[str, str]]  # (plot name, reason)
    missing: list[str]  # `dir` of the index, relative to the study directory
    plotted_runs: dict[str, Path]  # variant -> run directory


def aggregate_study(
    study: Study,
    formats: list[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
    plot_runs: bool = False,
) -> AggregateReport:
    """Write `runs.csv` (one row per run), `variants.csv` (statistics over seeds), the
    comparison plots, the galleries of representative runs and `summary.md` to
    `<study>/results/`, which is removed first.

    With `plot_runs`, the representative run of every variant (closest to the median of
    the test metric) also gets all plots of a single experiment in its `plots/`. Runs
    without artifacts are reported as missing; a study without any trained run is an error.
    """
    results = load_results(study)
    if not results.summaries:
        raise StudyError(
            f"no run of {study.directory.as_posix()} has artifacts: "
            f"run `compute train {study.directory.as_posix()}` first"
        )
    directory = study.directory / RESULTS_DIR
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()

    files = [directory / "runs.csv", directory / "variants.csv"]
    variants = variants_table(results)
    results.runs.write_csv(files[0])
    variants.write_csv(files[1])

    skipped = []
    with plt.rc_context(STYLE):
        for plot in STUDY_PLOTS:
            if not plot.applies(results):
                continue
            reason = plot.skip_reason(results)
            if reason is not None:
                skipped.append((plot.name, reason))
                continue
            files += save_figure(plot.draw(results), directory, plot.name, formats, dpi)
        reason = gallery_skip_reason(results)
        if reason is None:
            for name, figure in plot_study_gallery(results):
                files += save_figure(figure, directory, name, formats, dpi)
        else:
            skipped.append(("gallery", reason))

    plotted = representative_runs(results) if plot_runs else {}
    for run in plotted.values():
        plot_experiment(load_experiment(run), formats=formats, dpi=dpi)
    files.append(write_study_summary(results, variants, plotted, directory))
    return AggregateReport(
        directory=directory,
        files=files,
        skipped=skipped,
        missing=results.missing,
        plotted_runs=plotted,
    )
