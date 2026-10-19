"""Final compared metric (`split` of `[report]`, test by default) over two axes: rows are
the `color` levels, columns the `x` levels.

Cells show the mean and the standard deviation over seeds, or for errors (log color scale)
the median and the quartiles (see `center_and_spread`).
"""

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
    significant_places,
)
from analysis.plots.study_panels import (
    CELL_CHARACTER_CM,
    Metric,
    axes_width_cm,
    category_ticks,
    center_and_spread,
    finite,
    fits,
    levels,
    metric_of,
    panel_figure,
    panels,
    select,
)
from analysis.study import StudyResults
from analysis.study.run_summary import FINISHED
from analysis.style import GRID, INK

PLUS_MINUS = chr(0xB1)
EN_DASH = chr(0x2013)
# Font sizes of the cell text; the smaller one when the spread alone is wider than a cell.
CELL_FONT_SIZE = 8
SMALL_CELL_FONT_SIZE = 7
# Cells darker than this share of the color range get white text.
DARK_TEXT_THRESHOLD = 0.6


def heatmap_skip_reason(results: StudyResults) -> str | None:
    roles = results.study.roles
    if roles.x is None or roles.color is None:
        return "needs `x` and `color` in [report] of study.toml"
    return None


def plot_study_heatmap(results: StudyResults, column: str | None = None) -> Figure:
    """One heatmap per panel level; cells show the center and the spread over seeds.
    `column` selects another metric than the primary one (see `metric_of`)."""
    metric = metric_of(results, column)
    study = results.study
    study_panels = panels(results)
    x_levels = levels(results, "x")
    color_levels = levels(results, "color")
    # At least as tall as the colorbar label.
    height = min(10.0, max(7.0, 3.0 + 1.2 * len(color_levels)))
    figure, axes_list = panel_figure(len(study_panels), single_height_cm=height)
    tables = [_table(results, metric, panel.runs, x_levels, color_levels) for panel in study_panels]
    values = np.concatenate([finite(table[0].ravel()) for table in tables] or [np.array([])])
    log = metric.log
    norm = _norm(values, log)
    # Darker is better: high accuracy, low error.
    cmap = mpl.colormaps["Blues_r" if log else "Blues"].with_extremes(bad=GRID)
    for axes, panel, (center, low, high) in zip(axes_list, study_panels, tables, strict=True):
        image = axes.imshow(center, cmap=cmap, norm=norm, aspect="auto")
        _annotate(axes, center, low, high, norm, log)
        category_ticks(axes, [study.label(level) for level in x_levels])
        axes.set_yticks(range(len(color_levels)), [study.label(level) for level in color_levels])
        axes.set_xlabel(study.label(study.roles.x))
        axes.set_ylabel(study.label(study.roles.color))
        axes.grid(visible=False)
        axes.tick_params(length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
        if title := panel.title(study):
            axes.set_title(title)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, low and high end of the spread per cell over the finished runs (NaN if none;
    the spread is NaN for a single run)."""
    center = np.full((len(color_levels), len(x_levels)), np.nan)
    low = np.full_like(center, np.nan)
    high = np.full_like(center, np.nan)
    for row, color_level in enumerate(color_levels):
        for col, x in enumerate(x_levels):
            variant = select(runs, results, x, color_level)
            finished = variant.filter(pl.col("status").is_in(FINISHED))
            values = finite(finished[metric.column].cast(float).fill_null(np.nan).to_numpy())
            values = metric.scale * values
            if values.size:
                center[row, col], spread_low, spread_high = center_and_spread(values, metric.log)
            if values.size > 1:
                low[row, col], high[row, col] = spread_low, spread_high
    return center, low, high


def _norm(values: np.ndarray, log: bool) -> Normalize:
    if values.size == 0:
        return Normalize(0, 1)
    low, high = values.min(), values.max()
    if log and low > 0:
        return LogNorm(low, high if high > low else low * 10)
    return Normalize(low, high if high > low else low + 1)


def _annotate(
    axes: Axes,
    center: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    norm: Normalize,
    log: bool,
) -> None:
    """Center and spread in every cell: on one line, on two lines when one line is wider
    than a cell, in a smaller font when the spread alone is still too wide, and with the
    quartiles on lines of their own when even that does not fit (many columns)."""
    parts = {
        (row, col): _cell_parts(value, low[row, col], high[row, col], log)
        for (row, col), value in np.ndenumerate(center)
        if not np.isnan(value)
    }
    # The colorbar takes about a fifth of the panel.
    cell_cm = 0.8 * axes_width_cm(axes) / center.shape[1]
    texts = {cell: _one_line(main, spread, log) for cell, (main, spread) in parts.items()}
    font_size = CELL_FONT_SIZE
    if not fits(list(texts.values()), cell_cm, CELL_CHARACTER_CM):
        texts = {cell: _two_lines(main, spread, log) for cell, (main, spread) in parts.items()}
        lines = [line for text in texts.values() for line in text.splitlines()]
        if not fits(lines, cell_cm, CELL_CHARACTER_CM):
            font_size = SMALL_CELL_FONT_SIZE
            small_character_cm = CELL_CHARACTER_CM * SMALL_CELL_FONT_SIZE / CELL_FONT_SIZE
            if log and not fits(lines, cell_cm, small_character_cm):
                texts = {cell: _three_lines(main, spread) for cell, (main, spread) in parts.items()}
    for (row, col), value in np.ndenumerate(center):
        if np.isnan(value):
            text, color = DASH, INK
        else:
            text = texts[row, col]
            shade = float(norm(value))
            dark = (1 - shade if log else shade) > DARK_TEXT_THRESHOLD
            color = "white" if dark else INK
        axes.text(col, row, text, ha="center", va="center", fontsize=font_size, color=color)


def _cell_parts(center: float, low: float, high: float, log: bool) -> tuple[str, str | None]:
    """Center and spread of a cell: `97,2` and `0,4` (accuracy in percent: mean and
    deviation) or `0,153` and `0,045-0,200` (errors: median with 3 significant digits and
    the quartiles, no finer than the median). No spread for a single run."""
    if log:
        main = format_significant(center)
        if not np.isfinite(low):
            return main, None
        decimals = max(significant_places(center), 0)
        quartiles = (format_significant(value, max_decimals=decimals) for value in (low, high))
        return main, EN_DASH.join(quartiles)
    main = format_number(center, 1)
    return main, format_number(center - low, 1) if np.isfinite(low) else None


def _one_line(main: str, spread: str | None, log: bool) -> str:
    """`97,2 ± 0,4` or `0,153 (0,045-0,200)`."""
    if spread is None:
        return main
    return f"{main} ({spread})" if log else f"{main} {PLUS_MINUS} {spread}"


def _two_lines(main: str, spread: str | None, log: bool) -> str:
    """The spread on a line of its own; the quartiles without the parentheses."""
    if spread is None:
        return main
    return f"{main}\n{spread}" if log else f"{main}\n{PLUS_MINUS} {spread}"


def _three_lines(main: str, spread: str | None) -> str:
    """Errors only: the median, then each quartile on a line of its own (`0,045-` and
    `0,200`)."""
    if spread is None:
        return main
    low, high = spread.split(EN_DASH)
    return f"{main}\n{low}{EN_DASH}\n{high}"
