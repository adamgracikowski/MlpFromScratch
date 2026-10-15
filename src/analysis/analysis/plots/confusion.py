"""Confusion matrix of a classifier on one split."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from analysis import labels
from analysis.experiment import Experiment
from analysis.formatting import format_number, format_percent
from analysis.style import INK, new_figure

# Cells darker than this share of the row get white text.
WHITE_TEXT_ABOVE = 0.55


def confusion_matrix(
    true_labels: np.ndarray, predicted_labels: np.ndarray, classes: list[int]
) -> np.ndarray:
    """Counts `[i, j]` of samples of class `classes[i]` predicted as `classes[j]`."""
    index = {label: position for position, label in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true, predicted in zip(true_labels, predicted_labels, strict=True):
        counts[index[int(true)], index[int(predicted)]] += 1
    return counts


def row_shares(counts: np.ndarray) -> np.ndarray:
    """Share of every cell in its row (true class); 0 for a class without samples."""
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)


def plot_confusion(experiment: Experiment, split: str) -> Figure:
    """Counts and row percentages; the color shows the share of the true class."""
    classes = experiment.classes
    predictions = experiment.predictions[split]
    counts = confusion_matrix(predictions.labels, predictions.predicted, classes)
    shares = row_shares(counts)

    class_count = len(classes)
    side = min(14.0, 5.0 + 1.1 * class_count)  # with the colorbar at most 16 cm wide
    figure, axes = new_figure(side + 2.0, side)
    image = axes.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
    font_size = 9 if class_count <= 5 else 7
    for i in range(class_count):
        for j in range(class_count):
            color = "white" if shares[i, j] > WHITE_TEXT_ABOVE else INK
            axes.text(
                j,
                i,
                f"{counts[i, j]}\n{format_percent(shares[i, j])}",
                ha="center",
                va="center",
                color=color,
                fontsize=font_size,
            )

    ticks = np.arange(class_count)
    axes.set_xticks(ticks, [str(label) for label in classes])
    axes.set_yticks(ticks, [str(label) for label in classes])
    axes.set_xlabel(labels.PREDICTED_CLASS)
    axes.set_ylabel(labels.TRUE_CLASS)
    axes.xaxis.set_label_position("top")
    axes.xaxis.tick_top()
    axes.tick_params(length=0)
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.colorbar(
        image,
        ax=axes,
        label=labels.ROW_SHARE,
        format=FuncFormatter(lambda value, _: format_number(100 * value, 0)),
        fraction=0.05,
        pad=0.03,
    )
    return figure
