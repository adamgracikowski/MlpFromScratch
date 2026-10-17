"""Decision regions of a classifier in R^2 with the test samples."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analysis import labels
from analysis.experiment import Experiment
from analysis.series import evenly_spaced
from analysis.style import INK, MUTED, class_style, new_figure

# Opacity of the class color over white, from the least to the most confident prediction.
# The most confident regions stay pastel, so the samples on top remain readable.
REGION_ALPHA = (0.06, 0.40)
# Test samples drawn at most: tens of thousands of markers hide the regions and the boundaries.
MAX_POINTS = 3000


def plot_decision_boundary(experiment: Experiment) -> Figure:
    """Class regions shaded by confidence, their boundaries and at most `MAX_POINTS` evenly
    spaced test samples."""
    figure, axes = new_figure(16, 11)
    draw_decision_regions(axes, experiment, max_points=MAX_POINTS)
    axes.set_xlabel(labels.variable(experiment.features[0]))
    axes.set_ylabel(labels.variable(experiment.features[1]))
    axes.legend(
        handles=legend_handles(experiment.classes), loc="upper left", bbox_to_anchor=(1.02, 1)
    )
    return figure


def draw_decision_regions(
    axes: Axes, experiment: Experiment, max_points: int | None = None
) -> None:
    """Draw the class regions, their boundaries and the test samples, misclassified ones
    marked; with `max_points`, only that many evenly spaced test samples."""
    grid = experiment.grid
    x, y = grid.axes
    probabilities = grid.outputs  # (r, r, K), [i, j] is the point (x[j], y[i])
    classes = experiment.classes

    extent = [*_cell_bounds(x), *_cell_bounds(y)]
    axes.imshow(
        region_colors(probabilities),
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )
    for k in range(len(classes)):
        margin = class_margin(probabilities, k)
        if margin.min() < 0.0 < margin.max():  # the region of class k has a boundary
            axes.contour(x, y, margin, levels=[0.0], colors=INK, linewidths=0.8)

    test = experiment.predictions["test"]
    shown = evenly_spaced(len(test.rows), len(test.rows) if max_points is None else max_points)
    inputs, true, predicted = test.inputs[shown], test.labels[shown], test.predicted[shown]
    for index, label in enumerate(classes):
        color, marker = class_style(index)
        points = inputs[true == label]
        axes.scatter(
            points[:, 0],
            points[:, 1],
            s=12,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    wrong = predicted != true
    for index, label in enumerate(classes):
        color, _ = class_style(index)
        points = inputs[wrong & (true == label)]
        axes.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            color=color,
            marker="X",
            edgecolors=INK,
            linewidths=0.6,
            zorder=4,
        )

    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.grid(False)


def legend_handles(classes: list[int]) -> list[Line2D]:
    """Legend entries of the classes and of the misclassified samples."""
    handles = [
        Line2D(
            [],
            [],
            color=class_style(index)[0],
            marker=class_style(index)[1],
            linestyle="none",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=labels.class_name(label),
        )
        for index, label in enumerate(classes)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=MUTED,
            marker="X",
            linestyle="none",
            markersize=6,
            markeredgecolor=INK,
            markeredgewidth=0.6,
            label=labels.MISCLASSIFIED,
        )
    )
    return handles


def region_colors(probabilities: np.ndarray) -> np.ndarray:
    """RGB image: the color of the predicted class over white, stronger where confident.

    The confidence `max p` is rescaled from `[1/K, 1]` (uniform guess to certainty) to the
    opacity range `REGION_ALPHA`.
    """
    class_count = probabilities.shape[-1]
    predicted = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    chance = 1.0 / class_count
    strength = np.clip((confidence - chance) / (1.0 - chance), 0.0, 1.0)
    low, high = REGION_ALPHA
    alpha = (low + (high - low) * strength)[..., None]
    palette = np.array([to_rgb(class_style(k)[0]) for k in range(class_count)])
    return (1.0 - alpha) + alpha * palette[predicted]


def class_margin(probabilities: np.ndarray, k: int) -> np.ndarray:
    """`p_k - max_{j != k} p_j`: positive inside the region of class k, 0 on its boundary."""
    others = np.delete(probabilities, k, axis=-1)
    return probabilities[..., k] - others.max(axis=-1)


def _cell_bounds(axis: np.ndarray) -> tuple[float, float]:
    """Outer edges of the grid cells centered on the axis points."""
    step = (axis[-1] - axis[0]) / (len(axis) - 1) if len(axis) > 1 else 1.0
    return axis[0] - step / 2, axis[-1] + step / 2
