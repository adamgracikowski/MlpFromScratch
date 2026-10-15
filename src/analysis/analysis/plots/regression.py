"""Regression result: the network curve over the data, and its residuals on the test set."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from analysis import labels
from analysis.experiment import Experiment
from analysis.style import BLUE, GREEN, GRID, INK, VERMILLION, new_figure

# Training samples are drawn as points; the dense test set is a line.
RASTERIZE_ABOVE = 5000


def plot_regression(experiment: Experiment) -> Figure:
    """Data, target function and network output (top); test residuals (bottom)."""
    figure, (result, residuals) = new_figure(
        16, 13, rows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    predictions = experiment.predictions
    train = predictions["train"]
    test = predictions["test"]

    # Inputs outside the training range: there the network extrapolates.
    low, high = train.inputs[:, 0].min(), train.inputs[:, 0].max()
    grid_x = experiment.grid.axes[0]
    outside = [(start, end) for start, end in ((grid_x[0], low), (high, grid_x[-1])) if start < end]
    for axes in (result, residuals):
        for start, end in outside:
            axes.axvspan(start, end, color=GRID, alpha=0.6, linewidth=0, zorder=0)

    order = np.argsort(test.inputs[:, 0], kind="stable")
    test_x = test.inputs[order, 0]
    result.plot(
        test_x,
        test.targets[order],
        color=INK,
        linewidth=0.9,
        linestyle="--",
        label=labels.TARGET_FUNCTION,
        zorder=2,
    )
    result.scatter(
        train.inputs[:, 0],
        train.targets,
        s=10,
        color=BLUE,
        edgecolors="white",
        linewidths=0.3,
        label=labels.split_name("train"),
        zorder=3,
        rasterized=len(train.rows) > RASTERIZE_ABOVE,
    )
    if "validation" in predictions:
        validation = predictions["validation"]
        result.scatter(
            validation.inputs[:, 0],
            validation.targets,
            s=12,
            facecolors="none",
            edgecolors=GREEN,
            linewidths=0.6,
            label=labels.split_name("validation"),
            zorder=3,
        )
    result.plot(
        grid_x,
        experiment.grid.outputs[:, 0],
        color=VERMILLION,
        linewidth=1.5,
        label=labels.NETWORK_OUTPUT,
        zorder=4,
    )
    result.set_ylabel(labels.variable(experiment.metadata["data"]["target"]))
    handles, _ = result.get_legend_handles_labels()
    if outside:
        handles.append(Patch(color=GRID, alpha=0.6, label=labels.OUTSIDE_TRAINING_RANGE))
    result.legend(handles=handles, loc="best")

    residuals.axhline(0.0, color=INK, linewidth=0.6, zorder=1)
    residuals.plot(
        test_x,
        test.outputs[order, 0] - test.targets[order],
        color=VERMILLION,
        linewidth=1.0,
        zorder=2,
    )
    residuals.set_xlabel(labels.variable(experiment.features[0]))
    residuals.set_ylabel(labels.RESIDUAL)
    residuals.set_xlim(grid_x[0], grid_x[-1])
    return figure
