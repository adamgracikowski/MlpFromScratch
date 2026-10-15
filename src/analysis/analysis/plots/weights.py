"""Evolution of the weights: sampled parameters and the norms of every layer."""

import math

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analysis import labels
from analysis.experiment import Experiment
from analysis.formatting import DecimalCommaFormatter
from analysis.series import evenly_spaced, finite_or_nan
from analysis.style import BLUE, CATEGORICAL, INK, VERMILLION, new_figure

# Denser plots are rasterized in the PDF to keep it small.
RASTERIZE_ABOVE = 20000
# Snapshots drawn in the heatmap; more columns than pixels only slow drawing down.
HEATMAP_COLUMNS = 1500
# Percentile of |value| that ends the color scale, so a few outliers do not wash it out.
COLOR_LIMIT_PERCENTILE = 99.5


def plot_weight_trajectories(experiment: Experiment) -> Figure:
    """Values of the sampled weights and biases after every snapshot, one panel per layer."""
    weights = experiment.weights
    figure, panels = _layer_panels(weights.layer_count, panel_height=4.5)
    iterations = weights.iteration
    for layer, axes in enumerate(panels):
        values = finite_or_nan(weights.values[layer])
        is_bias = _is_bias(experiment, layer)
        dense = values.size > RASTERIZE_ABOVE
        for mask, color in ((~is_bias, BLUE), (is_bias, VERMILLION)):
            if mask.any():
                axes.plot(
                    iterations,
                    values[:, mask],
                    color=color,
                    linewidth=0.6,
                    alpha=0.6,
                    rasterized=dense,
                )
        axes.set_title(labels.layer_name(layer, weights.layer_count))
        axes.set_ylabel(labels.WEIGHT_VALUE)

    handles = [Line2D([], [], color=BLUE, label=labels.WEIGHTS)]
    if experiment.metadata["network"]["bias"]:
        handles.append(Line2D([], [], color=VERMILLION, label=labels.BIASES))
    figure.legend(handles=handles, loc="outside upper center", ncols=len(handles))
    return figure


def plot_weight_norms(experiment: Experiment) -> Figure:
    """Frobenius norm of W (solid) and Euclidean norm of b (dashed) of every layer."""
    weights = experiment.weights
    figure, axes = new_figure(16, 9)
    bias = experiment.metadata["network"]["bias"]
    for layer in range(weights.layer_count):
        color = CATEGORICAL[layer % len(CATEGORICAL)]
        name = labels.layer_name(layer, weights.layer_count)
        axes.plot(
            weights.iteration,
            finite_or_nan(weights.weight_norms[:, layer]),
            color=color,
            label=name,
        )
        if bias:
            axes.plot(
                weights.iteration,
                finite_or_nan(weights.bias_norms[:, layer]),
                color=color,
                linestyle="--",
            )
    axes.set_xlabel(labels.ITERATION)
    axes.set_ylabel(labels.NORM)

    handles, names = axes.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, label=labels.WEIGHT_NORM))
    if bias:
        handles.append(Line2D([], [], color=INK, linestyle="--", label=labels.BIAS_NORM))
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1))
    return figure


def plot_weight_heatmap(experiment: Experiment) -> Figure:
    """Sampled parameters (rows, sorted by final value) over the snapshots (columns)."""
    weights = experiment.weights
    figure, panels = _layer_panels(weights.layer_count, panel_height=4.5)
    columns = evenly_spaced(len(weights.iteration), HEATMAP_COLUMNS)
    iterations = weights.iteration[columns].astype(np.float64)
    x_edges = _edges(iterations)
    for layer, axes in enumerate(panels):
        values = finite_or_nan(weights.values[layer][columns])
        order = np.argsort(_final_values(values), kind="stable")
        image = np.ma.masked_invalid(values[:, order].T)
        limit = _color_limit(image)
        mesh = axes.pcolormesh(
            x_edges,
            np.arange(image.shape[0] + 1),
            image,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        figure.colorbar(
            mesh, ax=axes, label=labels.WEIGHT_VALUE, format=DecimalCommaFormatter(), pad=0.02
        )
        axes.set_title(labels.layer_name(layer, weights.layer_count))
        axes.set_ylabel(labels.SAMPLED_WEIGHTS)
        axes.set_yticks([])
        axes.grid(False)
    return figure


def _layer_panels(layer_count: int, panel_height: float) -> tuple[Figure, list[Axes]]:
    """One panel per layer: a single column up to 3 layers, then two columns.

    Panels share the iteration axis; only the lowest panel of a column shows its label.
    """
    columns = 1 if layer_count <= 3 else 2
    rows = math.ceil(layer_count / columns)
    figure, grid = new_figure(
        16, panel_height * rows + 1.0, rows=rows, columns=columns, squeeze=False, sharex=True
    )
    panels = list(grid.flat)
    for unused in panels[layer_count:]:
        unused.set_visible(False)
    for index, axes in enumerate(panels[:layer_count]):
        if index + columns < layer_count:
            axes.tick_params(labelbottom=False)
        else:
            axes.tick_params(labelbottom=True)
            axes.set_xlabel(labels.ITERATION)
    return figure, panels[:layer_count]


def _is_bias(experiment: Experiment, layer: int) -> np.ndarray:
    """Which sampled parameters of the layer are biases: row `in` of [W; b]."""
    fan_in = experiment.metadata["network"]["layer_sizes"][layer]
    return experiment.weights.indices[layer][:, 0] == fan_in


def _final_values(values: np.ndarray) -> np.ndarray:
    """Last finite value of every column (NaN columns sort last)."""
    final = np.full(values.shape[1], np.inf)
    for row in values:
        final = np.where(np.isnan(row), final, row)
    return final


def _color_limit(image: np.ma.MaskedArray) -> float:
    finite = np.abs(image.compressed())
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, COLOR_LIMIT_PERCENTILE))
    return limit if limit > 0 else 1.0


def _edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges around the given centers (midpoints, half a step past the ends)."""
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.concatenate(
        [
            [centers[0] - (middle[0] - centers[0])],
            middle,
            [centers[-1] + (centers[-1] - middle[-1])],
        ]
    )
