"""Learning curves of every variant: mean over seeds with a +-1 standard deviation band."""

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from analysis import labels
from analysis.plots.loss import fit_log_axis
from analysis.plots.study_panels import (
    color_of,
    is_log_metric,
    levels,
    line_style_of,
    metric_label,
    metric_scale,
    panel_figure,
    panels,
    primary_column,
    select,
    series_label,
)
from analysis.series import finite_or_nan
from analysis.study import StudyResults
from analysis.study.run_summary import FINISHED


def plot_study_curves(results: StudyResults) -> Figure:
    """The primary test metric (accuracy or MSE) after every epoch."""
    return _plot_curves(
        results,
        primary_column(results),
        metric_scale(results),
        metric_label(results),
        log=is_log_metric(results),
    )


def plot_study_train_loss(results: StudyResults) -> Figure:
    """The training loss after every epoch; all variants must use the same loss."""
    (loss,) = training_losses(results)
    return _plot_curves(results, "train_loss", 1.0, labels.train_loss(loss), log=True)


def training_losses(results: StudyResults) -> list[str]:
    """Distinct losses of the trained runs, in run order."""
    losses = (
        summary.metadata["config"]["training"]["loss"] for summary in results.summaries.values()
    )
    return list(dict.fromkeys(losses))


def mean_and_sd(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and sample standard deviation per epoch of series of different lengths (early
    stopping): each epoch uses the runs that reached it. Returns the epochs too; the
    deviation is NaN where fewer than two runs are left."""
    length = max(len(values) for values in series)
    table = np.full((len(series), length), np.nan)
    for row, values in enumerate(series):
        table[row, : len(values)] = values
    table[~np.isfinite(table)] = np.nan
    counts = np.sum(np.isfinite(table), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(table, axis=0) / counts
        sd = np.sqrt(np.nansum((table - mean) ** 2, axis=0) / (counts - 1))
    mean[counts == 0] = np.nan
    sd[counts < 2] = np.nan
    return np.arange(length), mean, sd


def _plot_curves(results: StudyResults, column: str, scale: float, label: str, log: bool) -> Figure:
    study_panels = panels(results)
    figure, axes_list = panel_figure(len(study_panels))
    for axes, panel in zip(axes_list, study_panels, strict=True):
        drawn = _draw_panel(axes, results, panel.runs, column, scale)
        if panel.title:
            axes.set_title(panel.title)
        axes.set_xlabel(labels.EPOCH)
        axes.set_ylabel(label)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        axes.set_xlim(left=0)
        if log:
            fit_log_axis(axes, drawn)
    handles, names = axes_list[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, names, loc="outside lower center", ncols=min(len(handles), 4))
    return figure


def _draw_panel(
    axes: Axes, results: StudyResults, runs: pl.DataFrame, column: str, scale: float
) -> list[np.ndarray]:
    drawn = []
    index = 0
    for x in levels(results, "x"):
        for color_level in levels(results, "color"):
            variant = select(runs, results, x, color_level)
            finished = variant.filter(pl.col("status").is_in(FINISHED)).sort("seed")
            series = [
                scale * finite_or_nan(results.summaries[run].epochs[column].to_numpy())
                for run in finished["dir"]
            ]
            if not series:
                continue
            epochs, mean, sd = mean_and_sd(series)
            color = color_of(index)
            axes.plot(
                epochs,
                mean,
                color=color,
                linestyle=line_style_of(index),
                label=series_label(x, color_level),
            )
            axes.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
            drawn += [mean - sd, mean + sd, mean]
            index += 1
    return drawn
