"""Loss and quality metrics during training."""

import math

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from analysis import labels
from analysis.experiment import SPLITS, Experiment
from analysis.series import finite_or_nan, moving_average
from analysis.style import BLUE, MUTED, SPLIT_COLORS, new_figure

# Denser lines are rasterized in the PDF to keep it small.
RASTERIZE_ABOVE = 2000
# Shorter runs get a marker at every epoch, so that even a single epoch is visible.
MARKERS_UP_TO_EPOCHS = 25
# Log-axis limit: a diverged run reaches ~1e300, where matplotlib's autoscaling and tick
# placement overflow. Larger values leave the plot at the top.
LOG_AXIS_MAX = 1e100


def plot_loss(experiment: Experiment) -> Figure:
    """Loss after every recorded window of iterations (top) and after every epoch (bottom)."""
    figure, (by_iteration, by_epoch) = new_figure(16, 14, rows=2)
    config = experiment.metadata["config"]
    loss_every = config["logging"]["loss_every"]
    history = experiment.history

    loss = finite_or_nan(history.loss)
    dense = len(loss) > RASTERIZE_ABOVE
    by_iteration.plot(
        history.iteration,
        loss,
        color=BLUE,
        alpha=0.3,
        linewidth=0.5,
        label=labels.loss_window(loss_every),
        rasterized=dense,
    )
    window = _windows_per_epoch(experiment)
    if window > 1:
        by_iteration.plot(
            history.iteration,
            moving_average(loss, window),
            color=BLUE,
            linewidth=1.2,
            label=labels.MOVING_AVERAGE,
            rasterized=dense,
        )
    by_iteration.set_xlabel(labels.ITERATION)
    by_iteration.set_xlim(left=0)
    epoch_losses = _plot_by_epoch(by_epoch, experiment, "loss")

    best = experiment.best_epoch
    if best is not None:
        epochs = experiment.epochs
        best_iteration = epochs.filter(epochs["epoch"] == best)["iteration"].item()
        _mark_best_epoch(by_iteration, best_iteration, best)
        _mark_best_epoch(by_epoch, best, best)

    loss_label = labels.loss_label(config["training"]["loss"])
    for axes, series in ((by_iteration, [loss]), (by_epoch, epoch_losses)):
        _fit_log_axis(axes, series)
        axes.set_ylabel(loss_label)
    # Noisy batch losses fill the whole panel: the legend gets a background.
    by_iteration.legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9
    )
    by_epoch.legend(loc="best")
    return figure


def plot_metrics(experiment: Experiment) -> Figure:
    """Accuracy (classification) or MSE in the original scale (regression) after every epoch."""
    figure, axes = new_figure(16, 9)
    if experiment.is_classification:
        _plot_by_epoch(axes, experiment, "accuracy", scale=100.0)
        axes.set_ylabel(labels.ACCURACY)
    else:
        _fit_log_axis(axes, _plot_by_epoch(axes, experiment, "mse"))
        axes.set_ylabel(labels.MSE)
    if experiment.best_epoch is not None:
        _mark_best_epoch(axes, experiment.best_epoch, experiment.best_epoch)
    axes.legend(loc="best")
    return figure


def _plot_by_epoch(
    axes: Axes, experiment: Experiment, metric: str, scale: float = 1.0
) -> list[np.ndarray]:
    """Draw `<split>_<metric>` of `epochs.csv` for every split; return the plotted series."""
    epochs = experiment.epochs
    epoch = epochs["epoch"].to_numpy()
    marker = "o" if len(epoch) <= MARKERS_UP_TO_EPOCHS + 1 else None
    series = []
    for split in _splits(experiment):
        values = scale * finite_or_nan(epochs[f"{split}_{metric}"].to_numpy())
        axes.plot(
            epoch,
            values,
            color=SPLIT_COLORS[split],
            marker=marker,
            markersize=3,
            label=labels.split_name(split),
        )
        series.append(values)
    axes.set_xlabel(labels.EPOCH)
    last = max(epoch.max(initial=0), 1)
    axes.set_xlim(-0.02 * last, 1.02 * last)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    return series


def _fit_log_axis(axes: Axes, series: list[np.ndarray]) -> None:
    """Switch to a log scale with limits from the positive finite values, 5% margin in decades.

    Matplotlib's own autoscaling overflows on values near the float limit (diverged runs).
    """
    with np.errstate(over="ignore"):
        axes.set_yscale("log")
    values = np.concatenate([np.ravel(values) for values in series]) if series else np.array([])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return
    low = math.log10(values.min())
    high = math.log10(min(values.max(), LOG_AXIS_MAX))
    margin = 0.05 * max(high - low, 0.1)
    top = min(high + margin, math.log10(LOG_AXIS_MAX))
    axes.set_ylim(10 ** (low - margin), 10**top)


def _splits(experiment: Experiment) -> list[str]:
    """Splits with metrics in `epochs.csv`, in the order train, validation, test."""
    return [split for split in SPLITS if f"{split}_loss" in experiment.epochs.columns]


def _windows_per_epoch(experiment: Experiment) -> int:
    """Number of recorded loss points per epoch: the window of the moving average."""
    result = experiment.metadata["result"]
    if result["epochs"] == 0:
        return 1
    iterations_per_epoch = result["iterations"] / result["epochs"]
    loss_every = experiment.metadata["config"]["logging"]["loss_every"]
    return max(1, int(np.ceil(iterations_per_epoch / loss_every)))


def _mark_best_epoch(axes: Axes, position: float, epoch: int) -> None:
    axes.axvline(
        position, color=MUTED, linestyle="--", linewidth=0.8, label=labels.best_epoch(epoch)
    )
