"""All plots of one experiment, written to `<experiment>/plots/`."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analysis.experiment import SPLITS, Experiment
from analysis.plots.confusion import plot_confusion
from analysis.plots.dataset import plot_dataset
from analysis.plots.decision_boundary import plot_decision_boundary
from analysis.plots.loss import plot_loss, plot_metrics
from analysis.plots.regression import plot_regression
from analysis.plots.weights import (
    plot_weight_heatmap,
    plot_weight_norms,
    plot_weight_trajectories,
)
from analysis.style import STYLE, save_figure
from analysis.summary import write_summary

PLOTS_DIR = "plots"
DEFAULT_FORMATS = ("pdf", "png")
DEFAULT_DPI = 200


@dataclass(frozen=True)
class Plot:
    """One plot: file name, drawing function and checks whether it fits the experiment.

    A plot that does not apply (another task, a split the run does not have) is left out
    silently; a plot with a skip reason (the data does not allow it) is reported.
    """

    name: str
    draw: Callable[[Experiment], Figure]
    applies: Callable[[Experiment], bool] = lambda _: True
    skip_reason: Callable[[Experiment], str | None] = lambda _: None


@dataclass(frozen=True)
class PlotReport:
    """Written files and the plots that did not fit the experiment."""

    directory: Path
    files: list[Path]
    skipped: list[tuple[str, str]]  # (plot name, reason)


def _input_space_reason(experiment: Experiment) -> str | None:
    """Plots of the input space need points in R^2 (classification) or R (regression)."""
    expected = 2 if experiment.is_classification else 1
    actual = len(experiment.features)
    if actual != expected:
        return f"needs {expected} input feature(s), the data has {actual}"
    return None


def _classification(experiment: Experiment) -> bool:
    return experiment.is_classification


def _regression(experiment: Experiment) -> bool:
    return not experiment.is_classification


def _confusion(split: str) -> Plot:
    return Plot(
        f"confusion_{split}",
        partial(plot_confusion, split=split),
        applies=lambda experiment: experiment.is_classification and split in experiment.predictions,
    )


PLOTS = [
    Plot("dataset", plot_dataset, skip_reason=_input_space_reason),
    Plot("decision_boundary", plot_decision_boundary, _classification, _input_space_reason),
    Plot("regression", plot_regression, _regression, _input_space_reason),
    Plot("loss", plot_loss),
    Plot("metrics", plot_metrics),
    Plot("weights_trajectories", plot_weight_trajectories),
    Plot("weight_norms", plot_weight_norms),
    Plot("weights_heatmap", plot_weight_heatmap),
    *(_confusion(split) for split in SPLITS),
]


def plot_experiment(
    experiment: Experiment, formats: list[str] = DEFAULT_FORMATS, dpi: int = DEFAULT_DPI
) -> PlotReport:
    """Remove `<experiment>/plots/`, write every plot that fits the experiment and the summary."""
    directory = experiment.directory / PLOTS_DIR
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()

    files = []
    skipped = []
    with plt.rc_context(STYLE):
        for plot in PLOTS:
            if not plot.applies(experiment):
                continue
            reason = plot.skip_reason(experiment)
            if reason is not None:
                skipped.append((plot.name, reason))
                continue
            files += save_figure(plot.draw(experiment), directory, plot.name, formats, dpi)
    files.append(write_summary(experiment, directory))
    return PlotReport(directory=directory, files=files, skipped=skipped)
