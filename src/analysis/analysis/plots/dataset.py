"""The dataset: training, validation and test samples, without the network."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analysis import labels
from analysis.experiment import Experiment
from analysis.style import MUTED, SPLIT_COLORS, class_style, new_figure

# Denser scatters are rasterized in the PDF to keep it small.
RASTERIZE_ABOVE = 5000

# Training samples are the main layer; test samples are a lighter background,
# validation samples are hollow.
_SPLIT_MARKS = {
    "train": {"s": 10, "linewidths": 0.3, "edgecolors": "white", "zorder": 3},
    "validation": {"s": 12, "linewidths": 0.6, "zorder": 4},
    "test": {"s": 6, "linewidths": 0, "alpha": 0.35, "zorder": 2},
}
_DRAW_ORDER = ("test", "train", "validation")


def plot_dataset(experiment: Experiment) -> Figure:
    """Samples of every split: classes in R^2, or y = f(x) for regression."""
    if experiment.is_classification:
        return _classification(experiment)
    return _regression(experiment)


def _classification(experiment: Experiment) -> Figure:
    figure, axes = new_figure(16, 11)
    classes = experiment.classes
    for split in _DRAW_ORDER:
        if split not in experiment.predictions:
            continue
        predictions = experiment.predictions[split]
        for index, label in enumerate(classes):
            color, marker = class_style(index)
            points = predictions.inputs[predictions.labels == label]
            _scatter(axes, points[:, 0], points[:, 1], split, color, marker)

    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(labels.variable(experiment.features[0]))
    axes.set_ylabel(labels.variable(experiment.features[1]))
    class_handles = [
        _handle(*class_style(index), label=labels.class_name(label))
        for index, label in enumerate(classes)
    ]
    split_handles = [
        _split_handle(split)
        for split in ("train", "validation", "test")
        if split in experiment.predictions
    ]
    axes.legend(handles=class_handles + split_handles, loc="upper left", bbox_to_anchor=(1.02, 1))
    return figure


def _regression(experiment: Experiment) -> Figure:
    figure, axes = new_figure(16, 10)
    for split in _DRAW_ORDER:
        if split not in experiment.predictions:
            continue
        predictions = experiment.predictions[split]
        _scatter(
            axes,
            predictions.inputs[:, 0],
            predictions.targets,
            split,
            SPLIT_COLORS[split],
            "o",
            label=labels.split_name(split),
        )

    axes.set_xlabel(labels.variable(experiment.features[0]))
    axes.set_ylabel(labels.variable(experiment.metadata["data"]["target"]))
    handles, names = axes.get_legend_handles_labels()
    order = [
        names.index(labels.split_name(s))
        for s in ("train", "validation", "test")
        if labels.split_name(s) in names
    ]
    axes.legend([handles[i] for i in order], [names[i] for i in order], markerscale=1.5)
    return figure


def _scatter(
    axes: Axes,
    x: np.ndarray,
    y: np.ndarray,
    split: str,
    color: str,
    marker: str,
    label: str | None = None,
) -> None:
    marks = dict(_SPLIT_MARKS[split])
    if split == "validation":
        marks |= {"facecolors": "none", "edgecolors": color}
    else:
        marks |= {"color": color}
    axes.scatter(x, y, marker=marker, label=label, rasterized=len(x) > RASTERIZE_ABOVE, **marks)


def _handle(color: str, marker: str, label: str) -> Line2D:
    return Line2D(
        [],
        [],
        color=color,
        marker=marker,
        linestyle="none",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.3,
        label=label,
    )


def _split_handle(split: str) -> Line2D:
    name = labels.split_name(split)
    if split == "train":
        return _handle(MUTED, "o", name)
    if split == "validation":
        return Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="none",
            markeredgewidth=0.6,
            label=name,
        )
    return Line2D(
        [],
        [],
        color=MUTED,
        marker="o",
        linestyle="none",
        markersize=3.5,
        alpha=0.35,
        markeredgewidth=0,
        label=name,
    )
