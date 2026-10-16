"""Mean final test metric over two axes: rows are the `color` levels, columns the `x` levels."""

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from analysis.formatting import (
    DASH,
    DecimalCommaFormatter,
    PlainLogFormatter,
    format_number,
    format_significant,
)
from analysis.plots.study_panels import (
    Metric,
    finite,
    levels,
    metric_of,
    panel_figure,
    panels,
    select,
)
from analysis.study import StudyResults
from analysis.study.run_summary import FINISHED
from analysis.style import GRID, INK

# Cells darker than this share of the color range get white text.
DARK_TEXT_THRESHOLD = 0.6


def heatmap_skip_reason(results: StudyResults) -> str | None:
    roles = results.study.roles
    if roles.x is None or roles.color is None:
        return "needs `x` and `color` in [report] of study.toml"
    return None


def plot_study_heatmap(results: StudyResults, column: str | None = None) -> Figure:
    """One heatmap per panel level; cells show the mean and the standard deviation.
    `column` selects another metric than the primary one (see `metric_of`)."""
    metric = metric_of(results, column)
    study_panels = panels(results)
    x_levels = levels(results, "x")
    color_levels = levels(results, "color")
    # At least as tall as the colorbar label.
    height = min(10.0, max(7.0, 3.0 + 1.2 * len(color_levels)))
    figure, axes_list = panel_figure(len(study_panels), single_height_cm=height)
    tables = [_table(results, metric, panel.runs, x_levels, color_levels) for panel in study_panels]
    values = np.concatenate([finite(mean.ravel()) for mean, _ in tables] or [np.array([])])
    log = metric.log
    norm = _norm(values, log)
    # Darker is better: high accuracy, low error.
    cmap = mpl.colormaps["Blues_r" if log else "Blues"].with_extremes(bad=GRID)
    for axes, panel, (mean, sd) in zip(axes_list, study_panels, tables, strict=True):
        image = axes.imshow(mean, cmap=cmap, norm=norm, aspect="auto")
        _annotate(axes, mean, sd, norm, log)
        axes.set_xticks(range(len(x_levels)), x_levels)
        axes.set_yticks(range(len(color_levels)), color_levels)
        axes.set_xlabel(results.study.roles.x)
        axes.set_ylabel(results.study.roles.color)
        axes.grid(visible=False)
        axes.tick_params(length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
        if panel.title:
            axes.set_title(panel.title)
        formatter = DecimalCommaFormatter(PlainLogFormatter() if log else None)
        colorbar = figure.colorbar(image, ax=axes, format=formatter, shrink=0.9)
        colorbar.set_label(metric.label)
        colorbar.outline.set_visible(False)
    return figure


def _table(
    results: StudyResults,
    metric: Metric,
    runs: pl.DataFrame,
    x_levels: list[str | None],
    color_levels: list[str | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation per cell over the finished runs (NaN if none)."""
    mean = np.full((len(color_levels), len(x_levels)), np.nan)
    sd = np.full_like(mean, np.nan)
    for row, color_level in enumerate(color_levels):
        for col, x in enumerate(x_levels):
            variant = select(runs, results, x, color_level)
            finished = variant.filter(pl.col("status").is_in(FINISHED))
            values = finite(finished[metric.column].cast(float).fill_null(np.nan).to_numpy())
            values = metric.scale * values
            if values.size:
                mean[row, col] = values.mean()
                sd[row, col] = values.std(ddof=1) if values.size > 1 else np.nan
    return mean, sd


def _norm(values: np.ndarray, log: bool) -> Normalize:
    if values.size == 0:
        return Normalize(0, 1)
    low, high = values.min(), values.max()
    if log and low > 0:
        return LogNorm(low, high if high > low else low * 10)
    return Normalize(low, high if high > low else low + 1)


def _annotate(axes: Axes, mean: np.ndarray, sd: np.ndarray, norm: Normalize, log: bool) -> None:
    for (row, col), value in np.ndenumerate(mean):
        if np.isnan(value):
            text, color = DASH, INK
        else:
            text = _cell_text(value, sd[row, col], log)
            shade = float(norm(value))
            dark = (1 - shade if log else shade) > DARK_TEXT_THRESHOLD
            color = "white" if dark else INK
        axes.text(col, row, text, ha="center", va="center", fontsize=8, color=color)


def _cell_text(mean: float, sd: float, log: bool) -> str:
    """`97,2 ± 0,4` (accuracy in percent) or `41 600 ± 2 470` (errors, 3 significant
    digits)."""
    if log:
        main = format_significant(mean)
        spread = format_significant(sd) if np.isfinite(sd) else None
    else:
        main = format_number(mean, 1)
        spread = format_number(sd, 1) if np.isfinite(sd) else None
    return main if spread is None else f"{main} {chr(0xB1)} {spread}"
