"""Results of the representative runs side by side: decision regions or regression curves.

One figure per combination of the levels of the axes other than `x` (`gallery`, or
`gallery_<level>__<level>`), with a panel per `x` level. Every panel shows the
representative run of its variant (`representative_runs`) and its test metric.
"""

import itertools
import math
from pathlib import Path

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis import labels
from analysis.experiment import Experiment, load_experiment
from analysis.formatting import format_percent, format_significant
from analysis.plots import decision_boundary, regression
from analysis.plots.decision_boundary import draw_decision_regions
from analysis.plots.regression import draw_regression_curve
from analysis.study import StudyResults, primary_metric, representative_runs
from analysis.style import MUTED, new_figure

COLUMNS = 3
# Test (classification) or training (regression) samples per panel: dense sets would
# cover the regions and the curve.
MAX_POINTS = 600
PANEL_HEIGHT_CM = {True: 5.2, False: 4.2}  # by `is_classification`
LEGEND_HEIGHT_CM = 1.2


def gallery_skip_reason(results: StudyResults) -> str | None:
    """The gallery draws the input space: R^2 (classification) or R (regression)."""
    expected = 2 if results.study.is_classification else 1
    for summary in results.summaries.values():
        actual = len(summary.metadata["data"]["features"])
        if actual != expected:
            return f"needs {expected} input feature(s), the data has {actual}"
    if not representative_runs(results):
        return "no variant has a finished run"
    return None


def plot_study_gallery(results: StudyResults) -> list[tuple[str, Figure]]:
    """One gallery figure per combination of the levels of the axes other than `x`."""
    study = results.study
    x_axis = study.roles.x
    others = [axis for axis in study.axes if axis.name != x_axis]
    x_levels = next((axis.levels for axis in study.axes if axis.name == x_axis), [None])
    representative = representative_runs(results)
    figures = []
    for combination in itertools.product(*(axis.levels for axis in others)):
        group = dict(zip((axis.name for axis in others), combination, strict=True))
        runs = [_run(results, representative, group | _x_level(x_axis, x)) for x in x_levels]
        if all(run is None for run in runs):
            continue
        name = f"gallery_{'__'.join(combination)}" if combination else "gallery"
        figures.append((name, _figure(results, x_levels, runs)))
    return figures


def _x_level(x_axis: str | None, level: str | None) -> dict[str, str]:
    return {} if x_axis is None else {x_axis: level}


def _run(
    results: StudyResults, representative: dict[str, Path], levels: dict[str, str]
) -> Path | None:
    """Representative run of the variant with the given levels; `None` without one."""
    runs = results.runs
    for name, level in levels.items():
        runs = runs.filter(pl.col(name) == level)
    if not runs.height:
        return None
    return representative.get(runs["variant"][0])


def _figure(results: StudyResults, x_levels: list[str | None], runs: list[Path | None]) -> Figure:
    classification = results.study.is_classification
    # Four panels as 2 x 2 rather than 3 + 1.
    columns = 2 if len(runs) == 4 else min(COLUMNS, len(runs))
    rows = math.ceil(len(runs) / columns)
    height = rows * PANEL_HEIGHT_CM[classification] + LEGEND_HEIGHT_CM
    figure, grid = new_figure(16, height, rows=rows, columns=columns, squeeze=False)
    flat = list(grid.ravel())
    for spare in flat[len(runs) :]:
        spare.set_visible(False)

    handles = None
    for index, (axes, level, run) in enumerate(zip(flat, x_levels, runs, strict=False)):
        if run is None:
            axes.set_axis_off()
            axes.set_title(labels.gallery_title(level, labels.NO_FINISHED_RUN), color=MUTED)
            continue
        experiment = load_experiment(run)
        if classification:
            draw_decision_regions(axes, experiment, max_points=MAX_POINTS)
            handles = decision_boundary.legend_handles(experiment.classes)
        else:
            outside = draw_regression_curve(axes, experiment, max_points=MAX_POINTS)
            handles = regression.legend_handles(axes, outside)
        axes.set_title(labels.gallery_title(level, _metric_text(results, run)))
        _label_outer(axes, experiment, index, columns, len(runs))

    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles) if classification else 2,
    )
    return figure


def _metric_text(results: StudyResults, run: Path) -> str:
    """Final test metric of a run: accuracy in percent or the MSE in the original scale."""
    summary = results.summaries[run.relative_to(results.study.directory).as_posix()]
    value = summary.metric("test", primary_metric(results.study))
    if results.study.is_classification:
        return format_percent(value)
    return labels.gallery_mse(format_significant(value))


def _label_outer(axes: Axes, experiment: Experiment, index: int, columns: int, count: int) -> None:
    """Axis labels on the panels of the left column and on the lowest panel of a column."""
    if index % columns == 0:
        if experiment.is_classification:
            vertical = experiment.features[1]
        else:
            vertical = experiment.metadata["data"]["target"]
        axes.set_ylabel(labels.variable(vertical))
    if index + columns >= count:
        axes.set_xlabel(labels.variable(experiment.features[0]))
