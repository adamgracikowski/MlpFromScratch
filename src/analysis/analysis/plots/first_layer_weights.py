"""Weights of the first layer as images: one map per neuron.

Every neuron of the first layer weighs the input pixels; reshaped to the image, its column of
`W` shows which pixels raise (red) or lower (blue) its input. Without hidden layers the first
layer is the output layer, and the maps are templates of the classes. Applies when the input
features are the pixels of square images (`Experiment.image_side`).

Pixels constant in the training set (e.g. the always blank border of MNIST) are normalized to
0, so their weights get no gradient and keep the random initial values; they are masked.
"""

import math

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from analysis import labels
from analysis.experiment import Experiment
from analysis.formatting import DecimalCommaFormatter
from analysis.style import GRID, new_figure

COLUMNS = 8
MAX_NEURONS = 32
PANEL_WIDTH_CM = 1.8
PANEL_HEIGHT_CM = 2.1  # the map and a one-line title


def plot_first_layer_weights(experiment: Experiment) -> Figure:
    """Columns of the trained `W` of layer 0 (the first `MAX_NEURONS` neurons) as maps on a
    shared symmetric color scale; pixels constant in the training set are masked."""
    side = experiment.image_side
    weights = experiment.model.weights[0]  # (d, out)
    count = min(MAX_NEURONS, weights.shape[1])
    train = experiment.predictions["train"].inputs
    constant = train.min(axis=0) == train.max(axis=0)
    shown = np.where(constant[:, None], np.nan, weights[:, :count])
    limit = float(np.nanmax(np.abs(shown))) if not constant.all() else 1.0
    cmap = mpl.colormaps["RdBu_r"].with_extremes(bad=GRID)
    output_layer = len(experiment.model.weights) == 1
    rows = math.ceil(count / COLUMNS)
    columns = min(COLUMNS, count)
    figure, grid = new_figure(
        columns * PANEL_WIDTH_CM + 1.8, rows * PANEL_HEIGHT_CM, rows, columns, squeeze=False
    )
    image = None
    for position, axes in enumerate(grid.flat):
        axes.set_axis_off()
        if position >= count:
            continue
        image = axes.imshow(
            shown[:, position].reshape(side, side),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        title = (
            labels.class_title(experiment.classes[position])
            if output_layer and experiment.classes is not None
            else labels.neuron_title(position + 1)
        )
        axes.set_title(title, fontsize=8)
    figure.colorbar(
        image,
        ax=grid,
        label=labels.WEIGHT_VALUE,
        format=DecimalCommaFormatter(),
        fraction=0.05,
        pad=0.02,
    )
    return figure
