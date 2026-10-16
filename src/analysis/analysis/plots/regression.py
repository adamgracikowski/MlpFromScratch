"""Regression result: the network curve over the data, and its residuals on the test set."""

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from analysis import labels
from analysis.experiment import Experiment
from analysis.series import evenly_spaced
from analysis.style import BLUE, GREEN, GRID, INK, VERMILLION, new_figure

# Training samples are drawn as points; the dense test set is a line.
RASTERIZE_ABOVE = 5000


def plot_regression(experiment: Experiment) -> Figure:
    """Data, target function and network output (top); test residuals (bottom)."""
    figure, (result, residuals) = new_figure(
        16, 13, rows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    outside = draw_regression_curve(result, experiment)
    shade_outside(residuals, outside)
    result.set_ylabel(labels.variable(experiment.metadata["data"]["target"]))
    result.legend(handles=legend_handles(result, outside), loc="best")

    test = experiment.predictions["test"]
    order = np.argsort(test.inputs[:, 0], kind="stable")
    grid_x = experiment.grid.axes[0]
    residuals.axhline(0.0, color=INK, linewidth=0.6, zorder=1)
    residuals.plot(
        test.inputs[order, 0],
        test.outputs[order, 0] - test.targets[order],
        color=VERMILLION,
        linewidth=1.0,
        zorder=2,
    )
    residuals.set_xlabel(labels.variable(experiment.features[0]))
    residuals.set_ylabel(labels.RESIDUAL)
    residuals.set_xlim(grid_x[0], grid_x[-1])
    return figure


def draw_regression_curve(
    axes: Axes, experiment: Experiment, max_points: int | None = None
) -> list[tuple[float, float]]:
    """Draw the target function (test set), the training and validation samples and the
    network output, with the inputs outside the training range shaded; with `max_points`,
    only that many evenly spaced training samples. Returns the shaded ranges."""
    predictions = experiment.predictions
    train = predictions["train"]
    test = predictions["test"]

    # Inputs outside the training range: there the network extrapolates.
    low, high = train.inputs[:, 0].min(), train.inputs[:, 0].max()
    grid_x = experiment.grid.axes[0]
    outside = [(start, end) for start, end in ((grid_x[0], low), (high, grid_x[-1])) if start < end]
    shade_outside(axes, outside)

    order = np.argsort(test.inputs[:, 0], kind="stable")
    axes.plot(
        test.inputs[order, 0],
        test.targets[order],
        color=INK,
        linewidth=0.9,
        linestyle="--",
        label=labels.TARGET_FUNCTION,
        zorder=2,
    )
    shown = evenly_spaced(len(train.rows), len(train.rows) if max_points is None else max_points)
    axes.scatter(
        train.inputs[shown, 0],
        train.targets[shown],
        s=10,
        color=BLUE,
        edgecolors="white",
        linewidths=0.3,
        label=labels.split_name("train"),
        zorder=3,
        rasterized=len(shown) > RASTERIZE_ABOVE,
    )
    if "validation" in predictions:
        validation = predictions["validation"]
        axes.scatter(
            validation.inputs[:, 0],
            validation.targets,
            s=12,
            facecolors="none",
            edgecolors=GREEN,
            linewidths=0.6,
            label=labels.split_name("validation"),
            zorder=3,
        )
    axes.plot(
        grid_x,
        experiment.grid.outputs[:, 0],
        color=VERMILLION,
        linewidth=1.5,
        label=labels.NETWORK_OUTPUT,
        zorder=4,
    )
    return outside


def shade_outside(axes: Axes, outside: list[tuple[float, float]]) -> None:
    for start, end in outside:
        axes.axvspan(start, end, color=GRID, alpha=0.6, linewidth=0, zorder=0)


def legend_handles(axes: Axes, outside: list[tuple[float, float]]) -> list[Artist]:
    """Legend entries of the drawn series, plus the shading when any range is shaded."""
    handles, _ = axes.get_legend_handles_labels()
    if outside:
        handles.append(Patch(color=GRID, alpha=0.6, label=labels.OUTSIDE_TRAINING_RANGE))
    return handles
