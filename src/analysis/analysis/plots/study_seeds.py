"""Final test metric of every variant: mean with +-1 standard deviation and every seed."""

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analysis import labels
from analysis.plots.loss import fit_log_axis
from analysis.plots.study_panels import (
    color_of,
    finite,
    is_log_metric,
    levels,
    metric_label,
    metric_scale,
    panel_figure,
    panels,
    primary_column,
    select,
)
from analysis.study import StudyResults
from analysis.study.run_summary import FINISHED
from analysis.style import BLUE, INK, MUTED

GROUP_WIDTH = 0.8  # share of a category taken by its color groups
SEED_SPREAD = 0.5  # share of a color group taken by the seed dots


def plot_study_seeds(results: StudyResults) -> Figure:
    """One panel per panel level: categories of the `x` axis, colors of the `color` axis."""
    study_panels = panels(results)
    figure, axes_list = panel_figure(len(study_panels))
    x_levels = levels(results, "x")
    color_levels = levels(results, "color")
    for axes, panel in zip(axes_list, study_panels, strict=True):
        values = _draw_panel(axes, results, panel.runs, x_levels, color_levels)
        if panel.title:
            axes.set_title(panel.title)
        axes.set_xticks(range(len(x_levels)), [level or "" for level in x_levels])
        axes.set_xlim(-0.5, len(x_levels) - 0.5)
        axes.grid(axis="x", visible=False)
        axes.set_xlabel(results.study.roles.x or "")
        axes.set_ylabel(metric_label(results))
        if is_log_metric(results):
            fit_log_axis(axes, [values])
    _legend(figure, results, color_levels)
    return figure


def _draw_panel(
    axes: Axes,
    results: StudyResults,
    runs: pl.DataFrame,
    x_levels: list[str | None],
    color_levels: list[str | None],
) -> np.ndarray:
    """Draw every variant of the panel; return the drawn values (for the log axis)."""
    column = primary_column(results)
    scale = metric_scale(results)
    group_width = GROUP_WIDTH / len(color_levels)
    drawn = []
    for position, x in enumerate(x_levels):
        for group, color_level in enumerate(color_levels):
            variant = select(runs, results, x, color_level)
            if variant.height == 0:
                continue
            center = position + (group - (len(color_levels) - 1) / 2) * group_width
            color = color_of(group) if results.study.roles.color else BLUE
            finished = variant.filter(pl.col("status").is_in(FINISHED)).sort("seed")
            values = scale * finished[column].cast(float).fill_null(np.nan).to_numpy()
            values = finite(values)
            drawn.append(values)
            if values.size:
                spread = SEED_SPREAD * group_width / 2
                offsets = np.linspace(-spread, spread, values.size) if values.size > 1 else [0]
                axes.scatter(
                    center + np.asarray(offsets),
                    values,
                    s=9,
                    color=color,
                    alpha=0.45,
                    linewidths=0,
                    zorder=2,
                )
                sd = values.std(ddof=1) if values.size > 1 else 0.0
                axes.errorbar(
                    center,
                    values.mean(),
                    yerr=sd,
                    fmt="o",
                    color=color,
                    markeredgecolor=INK,
                    markeredgewidth=0.6,
                    markersize=5,
                    ecolor=INK,
                    elinewidth=0.9,
                    capsize=3,
                    zorder=3,
                )
            diverged = variant.filter(pl.col("status") == "diverged").height
            if diverged:
                axes.annotate(
                    labels.diverged_runs(diverged, variant.height),
                    (center, 1.0),
                    xycoords=axes.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=MUTED,
                )
    return np.concatenate(drawn) if drawn else np.array([])


def _legend(figure: Figure, results: StudyResults, color_levels: list[str | None]) -> None:
    """Markers of the mean and of a seed, then the colors of the `color` axis."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=INK,
            markerfacecolor=MUTED,
            linestyle="-",
            markersize=5,
            label=labels.MEAN_AND_SD,
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=MUTED,
            alpha=0.45,
            linestyle="none",
            markersize=3,
            label=labels.SINGLE_SEED,
        ),
    ]
    if results.study.roles.color:
        handles += [
            Line2D(
                [],
                [],
                marker="s",
                color=color_of(index),
                linestyle="none",
                markersize=6,
                label=level,
            )
            for index, level in enumerate(color_levels)
        ]
    figure.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 4))
