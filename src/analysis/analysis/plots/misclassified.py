"""Misclassified test images, the most confident mistakes first.

Applies when the input features are the pixels of square images (`Experiment.image_side`).
"""

import math

import numpy as np
from matplotlib.figure import Figure

from analysis import labels
from analysis.experiment import Experiment
from analysis.formatting import format_percent
from analysis.style import new_figure

COLUMNS = 8
MAX_IMAGES = 32
PANEL_WIDTH_CM = 2.0
PANEL_HEIGHT_CM = 2.6  # the image and a two-line title


def misclassified_order(
    labels_true: np.ndarray, predicted: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """Indices of the misclassified samples, ordered by the probability the network gave to
    its (wrong) prediction, highest first; ties keep the sample order."""
    wrong = np.flatnonzero(labels_true != predicted)
    confidence = outputs[wrong].max(axis=1)
    return wrong[np.argsort(-confidence, kind="stable")]


def misclassified_reason(experiment: Experiment) -> str | None:
    """Reason to skip the plot: no test sample is misclassified."""
    test = experiment.predictions["test"]
    if not np.any(test.labels != test.predicted):
        return "every test sample is classified correctly"
    return None


def plot_misclassified(experiment: Experiment) -> Figure:
    """Up to `MAX_IMAGES` misclassified test images in gray, each with the true and the
    predicted class and the probability of the prediction."""
    side = experiment.image_side
    test = experiment.predictions["test"]
    chosen = misclassified_order(test.labels, test.predicted, test.outputs)[:MAX_IMAGES]
    rows = math.ceil(len(chosen) / COLUMNS)
    figure, grid = new_figure(
        COLUMNS * PANEL_WIDTH_CM, rows * PANEL_HEIGHT_CM, rows, COLUMNS, squeeze=False
    )
    # One gray scale for all images: the range of the pixels of the test set.
    low, high = float(test.inputs.min()), float(test.inputs.max())
    for position, axes in enumerate(grid.flat):
        axes.set_axis_off()
        if position >= len(chosen):
            continue
        sample = chosen[position]
        axes.imshow(
            test.inputs[sample].reshape(side, side),
            cmap="gray_r",
            vmin=low,
            vmax=high,
            interpolation="nearest",
        )
        axes.set_title(
            labels.misclassified_title(
                int(test.labels[sample]),
                int(test.predicted[sample]),
                format_percent(float(test.outputs[sample].max())),
            ),
            fontsize=8,
        )
    return figure
