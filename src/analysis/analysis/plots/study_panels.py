"""Layout shared by the comparison plots of a study: panels, categories and the metric.

The roles of the axes (`[report]` of `study.toml`) decide the layout: one panel per
combination of the levels of the `panels` axes, the levels of the `x` axis as categories
and the levels of the `color` axis as colors within a category.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis import labels
from analysis.study import StudyResults, primary_metric
from analysis.study.range_errors import INSIDE, OUTSIDE
from analysis.style import CATEGORICAL, CM, new_figure


@dataclass(frozen=True)
class Panel:
    """Runs of one combination of the panel levels."""

    levels: dict[str, str]  # panel axis -> level
    runs: pl.DataFrame

    @property
    def title(self) -> str | None:
        return labels.panel_title(self.levels) if self.levels else None


def panels(results: StudyResults) -> list[Panel]:
    """One panel per combination of the levels of the panel axes, in level order."""
    study = results.study
    axes = {axis.name: axis.levels for axis in study.axes}
    names = study.roles.panels
    found = []
    for combination in itertools.product(*(axes[name] for name in names)):
        levels = dict(zip(names, combination, strict=True))
        runs = results.runs
        for name, level in levels.items():
            runs = runs.filter(pl.col(name) == level)
        if runs.height:
            found.append(Panel(levels, runs))
    return found


def levels(results: StudyResults, role: str) -> list[str | None]:
    """Levels of the axis with the given role (`x` or `color`); `[None]` without one."""
    name = getattr(results.study.roles, role)
    if name is None:
        return [None]
    return next(axis.levels for axis in results.study.axes if axis.name == name)


def select(runs: pl.DataFrame, results: StudyResults, x: str | None, color: str | None):
    """Runs of one category and color within a panel."""
    roles = results.study.roles
    for name, level in ((roles.x, x), (roles.color, color)):
        if name is not None:
            runs = runs.filter(pl.col(name) == level)
    return runs


def series_label(x: str | None, color: str | None) -> str | None:
    """Legend entry of a variant within a panel: its `x` and `color` levels."""
    parts = [level for level in (x, color) if level is not None]
    return ", ".join(parts) if parts else None


def color_of(index: int) -> str:
    return CATEGORICAL[index % len(CATEGORICAL)]


def line_style_of(index: int) -> str:
    """Line style that tells apart series whose colors repeat (more than 8 series)."""
    return ("-", "--", ":", "-.")[(index // len(CATEGORICAL)) % 4]


# Rough width of a character of the 9 pt tick labels and of the 8 pt heatmap cells.
TICK_CHARACTER_CM = 0.17
CELL_CHARACTER_CM = 0.15


def axes_width_cm(axes: Axes) -> float:
    """Width of the axes in the figure grid (before the constrained layout shrinks it)."""
    return axes.get_position().width * axes.get_figure().get_figwidth() / CM


def fits(texts: list[str], slot_cm: float, character_cm: float) -> bool:
    """Whether the longest text fits a slot, with a margin of 10 %."""
    longest = max((len(text) for text in texts), default=0)
    return longest * character_cm <= 0.9 * slot_cm


def category_ticks(axes: Axes, names: list[str]) -> None:
    """One x tick per category; labels too long for their slot are slanted."""
    axes.set_xticks(range(len(names)), names)
    if names and not fits(names, axes_width_cm(axes) / len(names), TICK_CHARACTER_CM):
        axes.tick_params(axis="x", labelrotation=30)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def panel_figure(count: int, single_height_cm: float = 9.0) -> tuple[Figure, list[Axes]]:
    """A figure 16 cm wide with `count` panels in up to two columns; spare axes hidden."""
    columns = 1 if count == 1 else 2
    rows = math.ceil(count / columns)
    height = single_height_cm if count == 1 else 6.5 * rows
    figure, axes = new_figure(16, height, rows=rows, columns=columns, squeeze=False)
    flat = list(axes.ravel())
    for spare in flat[count:]:
        spare.set_visible(False)
    return figure, flat[:count]


def is_log_metric(results: StudyResults) -> bool:
    """Regression errors span orders of magnitude: they are drawn on a log axis."""
    return not results.study.is_classification


def metric_scale(results: StudyResults) -> float:
    """Accuracy is shown in percent."""
    return 100.0 if results.study.is_classification else 1.0


def metric_label(results: StudyResults) -> str:
    return labels.TEST_ACCURACY if results.study.is_classification else labels.TEST_MSE


def primary_column(results: StudyResults) -> str:
    return f"test_{primary_metric(results.study)}"


@dataclass(frozen=True)
class Metric:
    """A final metric compared across variants: column of the runs table, axis label,
    factor of the shown values and whether the axis is logarithmic."""

    column: str
    label: str
    scale: float
    log: bool


def metric_of(results: StudyResults, column: str | None = None) -> Metric:
    """The primary test metric, or the test MSE inside or outside the training range."""
    if column is None:
        return Metric(
            primary_column(results),
            metric_label(results),
            metric_scale(results),
            is_log_metric(results),
        )
    label = {INSIDE: labels.TEST_MSE_INSIDE, OUTSIDE: labels.TEST_MSE_OUTSIDE}[column]
    return Metric(column, label, 1.0, True)


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def center_and_spread(values: np.ndarray, log: bool) -> tuple[float, float, float]:
    """Center and spread of a metric over seeds as `(center, low, high)`: the mean and
    mean +- sample standard deviation, or for metrics on a log axis (errors) the median and
    the quartiles: errors over seeds have a long tail, the mean follows a few bad seeds and
    mean - sd is often negative. The spread is the center itself for a single value."""
    if log:
        low, center, high = np.percentile(values, [25, 50, 75])
        return float(center), float(low), float(high)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean, mean - sd, mean + sd


def spread_label(log: bool) -> str:
    """Legend entry of the center with its spread (see `center_and_spread`)."""
    return labels.MEDIAN_AND_QUARTILES if log else labels.MEAN_AND_SD
