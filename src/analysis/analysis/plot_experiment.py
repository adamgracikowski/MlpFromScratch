"""All plots of one experiment, written to `<experiment>/plots/`."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analysis.experiment import Experiment
from analysis.plots.dataset import plot_dataset
from analysis.plots.loss import plot_loss, plot_metrics
from analysis.plots.weights import (
    plot_weight_heatmap,
    plot_weight_norms,
    plot_weight_trajectories,
)
from analysis.style import STYLE, save_figure

PLOTS_DIR = "plots"
DEFAULT_FORMATS = ("pdf", "png")
DEFAULT_DPI = 200


@dataclass(frozen=True)
class Plot:
    """One plot: file name, drawing function and a check whether it fits the experiment."""

    name: str
    draw: Callable[[Experiment], Figure]
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


PLOTS = [
    Plot("dataset", plot_dataset, _input_space_reason),
    Plot("loss", plot_loss),
    Plot("metrics", plot_metrics),
    Plot("weights_trajectories", plot_weight_trajectories),
    Plot("weight_norms", plot_weight_norms),
    Plot("weights_heatmap", plot_weight_heatmap),
]


def plot_experiment(
    experiment: Experiment, formats: list[str] = DEFAULT_FORMATS, dpi: int = DEFAULT_DPI
) -> PlotReport:
    """Remove `<experiment>/plots/` and write every plot that fits the experiment into it."""
    directory = experiment.directory / PLOTS_DIR
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()

    files = []
    skipped = []
    with plt.rc_context(STYLE):
        for plot in PLOTS:
            reason = plot.skip_reason(experiment)
            if reason is not None:
                skipped.append((plot.name, reason))
                continue
            files += save_figure(plot.draw(experiment), directory, plot.name, formats, dpi)
    return PlotReport(directory=directory, files=files, skipped=skipped)
