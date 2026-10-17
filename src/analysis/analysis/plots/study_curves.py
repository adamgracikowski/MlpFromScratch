"""Learning curves of every variant: mean over seeds with a +-1 standard deviation band, or
for metrics on a log axis (errors, losses) the median with a band between the quartiles
(see `center_and_spread`).

Unlike the other comparison plots, the levels of the `color` axis get panels of their own
(next to the `panels` axes), so a panel holds one curve per `x` level: levels of `color`
often differ in the number of epochs (e.g. training set sizes), and a panel with every
combination would be unreadable.
"""

import math

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from analysis import labels
from analysis.plots.loss import fit_log_axis
from analysis.plots.study_panels import (
    Panel,
    color_of,
    is_log_metric,
    levels,
    line_style_of,
    metric_label,
    metric_scale,
    panel_figure,
    panels,
    primary_column,
    series_label,
)
from analysis.series import finite_or_nan
from analysis.study import StudyResults
from analysis.study.run_summary import FINISHED

# Share of the epochs left out when a linear axis is fitted (see `fit_after_warmup`).
WARMUP_SHARE = 0.1


def plot_study_curves(results: StudyResults) -> Figure:
    """The primary test metric (accuracy or MSE) after every epoch."""
    log = is_log_metric(results)
    return _plot_curves(
        results,
        primary_column(results),
        metric_scale(results),
        metric_label(results),
        log=log,
        bounds=None if log else (0.0, 100.0),
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
    table, counts = _epoch_table(series)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(table, axis=0) / counts
        sd = np.sqrt(np.nansum((table - mean) ** 2, axis=0) / (counts - 1))
    mean[counts == 0] = np.nan
    sd[counts < 2] = np.nan
    return np.arange(table.shape[1]), mean, sd


def median_and_quartiles(
    series: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and quartiles per epoch, like `mean_and_sd`; the quartiles are NaN where
    fewer than two runs are left."""
    table, counts = _epoch_table(series)
    quartiles = np.full((3, table.shape[1]), np.nan)
    reached = counts > 0
    quartiles[:, reached] = np.nanpercentile(table[:, reached], [25, 50, 75], axis=0)
    low, median, high = quartiles
    low[counts < 2] = np.nan
    high[counts < 2] = np.nan
    return np.arange(table.shape[1]), median, low, high


def _epoch_table(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Series as rows of a table padded with NaN (non-finite values too), and the number
    of values per epoch."""
    length = max(len(values) for values in series)
    table = np.full((len(series), length), np.nan)
    for row, values in enumerate(series):
        table[row, : len(values)] = values
    table[~np.isfinite(table)] = np.nan
    return table, np.sum(np.isfinite(table), axis=0)


def curve_panels(results: StudyResults) -> list[Panel]:
    """One panel per combination of the `panels` levels and the `color` level."""
    color = results.study.roles.color
    found = []
    for panel in panels(results):
        for level in levels(results, "color"):
            if color is None:
                found.append(panel)
                continue
            runs = panel.runs.filter(pl.col(color) == level)
            if runs.height:
                found.append(Panel(panel.levels | {color: level}, runs))
    return found


def _plot_curves(
    results: StudyResults,
    column: str,
    scale: float,
    label: str,
    log: bool,
    bounds: tuple[float, float] | None = None,
) -> Figure:
    """`bounds` are the possible values of the metric (accuracy: 0 to 100 %): the deviation
    bands are clipped to them, and the linear axis is fitted to the curves after the first
    epochs."""
    study_panels = curve_panels(results)
    figure, axes_list = panel_figure(len(study_panels))
    for axes, panel in zip(axes_list, study_panels, strict=True):
        drawn = _draw_panel(axes, results, panel.runs, column, scale, log, bounds)
        if title := panel.title(results.study):
            axes.set_title(title)
        axes.set_xlabel(labels.EPOCH)
        axes.set_ylabel(label)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        axes.set_xlim(left=0)
        if log:
            fit_log_axis(axes, drawn)
        else:
            fit_after_warmup(axes, drawn)
    handles, names = _legend_entries(axes_list)
    if handles:
        figure.legend(handles, names, loc="outside lower center", ncols=min(len(handles), 4))
    return figure


def fit_after_warmup(axes: Axes, drawn: list[np.ndarray]) -> None:
    """Fit the y axis to the curves without their first epochs: accuracy climbs from the
    chance level within a few epochs, and the climb would squeeze the differences between
    the variants into a thin strip."""
    tails = [values[math.ceil(WARMUP_SHARE * (len(values) - 1)) :] for values in drawn]
    values = np.concatenate([values[np.isfinite(values)] for values in tails] or [[]])
    if values.size == 0:
        return
    low, high = values.min(), values.max()
    margin = 0.05 * (high - low) if high > low else 1.0
    axes.set_ylim(low - margin, high + margin)


def _legend_entries(axes_list: list[Axes]) -> tuple[list, list[str]]:
    """Entries of all panels, each `x` level once, in level order."""
    entries = {}
    for axes in axes_list:
        for handle, name in zip(*axes.get_legend_handles_labels(), strict=True):
            entries.setdefault(name, handle)
    return list(entries.values()), list(entries)


def _draw_panel(
    axes: Axes,
    results: StudyResults,
    runs: pl.DataFrame,
    column: str,
    scale: float,
    log: bool,
    bounds: tuple[float, float] | None,
) -> list[np.ndarray]:
    """One curve per `x` level; a level keeps its color and line style in every panel."""
    drawn = []
    x_axis = results.study.roles.x
    for index, x in enumerate(levels(results, "x")):
        variant = runs if x_axis is None else runs.filter(pl.col(x_axis) == x)
        finished = variant.filter(pl.col("status").is_in(FINISHED)).sort("seed")
        series = [
            scale * finite_or_nan(results.summaries[run].epochs[column].to_numpy())
            for run in finished["dir"]
        ]
        if not series:
            continue
        if log:
            epochs, center, lower, upper = median_and_quartiles(series)
        else:
            epochs, center, sd = mean_and_sd(series)
            lower, upper = center - sd, center + sd
        color = color_of(index)
        axes.plot(
            epochs,
            center,
            color=color,
            linestyle=line_style_of(index),
            label=series_label(results.study.label(x), None),
        )
        if bounds is not None:
            lower, upper = np.clip(lower, *bounds), np.clip(upper, *bounds)
        axes.fill_between(epochs, lower, upper, color=color, alpha=0.18, linewidth=0)
        drawn += [lower, upper, center]
    return drawn
