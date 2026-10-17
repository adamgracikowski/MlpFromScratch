import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis.plots.study_curves import (
    curve_panels,
    fit_after_warmup,
    mean_and_sd,
    median_and_quartiles,
    plot_study_curves,
)
from analysis.study import load_results, load_study

REPORT = '[report]\nx = "activation"\ncolor = "dataset"'


def test_mean_and_sd_of_series_with_different_lengths() -> None:
    epochs, mean, sd = mean_and_sd([np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0])])

    assert epochs.tolist() == [0, 1, 2]
    assert mean.tolist() == [2.0, 3.0, 3.0]
    assert sd[0] == sd[1] == math.sqrt(2)
    assert math.isnan(sd[2])


def test_mean_and_sd_ignore_missing_values() -> None:
    _, mean, sd = mean_and_sd([np.array([1.0, np.nan]), np.array([3.0, np.nan])])

    assert mean[0] == 2.0
    assert math.isnan(mean[1])
    assert math.isnan(sd[1])


def test_median_and_quartiles_of_series_with_different_lengths() -> None:
    series = [np.array([1.0, 5.0]), np.array([2.0, np.inf]), np.array([3.0])]

    epochs, median, low, high = median_and_quartiles(series)

    assert epochs.tolist() == [0, 1]
    assert median.tolist() == [2.0, 5.0]
    assert (low[0], high[0]) == (1.5, 2.5)
    assert math.isnan(low[1])
    assert math.isnan(high[1])


def test_panel_per_color_level(make_study) -> None:
    results = load_results(load_study(make_study(report=REPORT)))

    found = curve_panels(results)

    assert [panel.title(results.study) for panel in found] == ["dataset: a", "dataset: b"]
    assert all(panel.runs.height == 6 for panel in found)


def test_panels_and_color_combined(make_study) -> None:
    axes = {"activation": ["sigmoid", "tanh"], "dataset": ["a", "b"], "size": ["1", "2"]}
    report = '[report]\nx = "activation"\ncolor = "size"\npanels = ["dataset"]'
    results = load_results(load_study(make_study(axes=axes, seeds=(1, 2), report=report)))

    titles = [panel.title(results.study) for panel in curve_panels(results)]

    assert titles == [
        "dataset: a, size: 1",
        "dataset: a, size: 2",
        "dataset: b, size: 1",
        "dataset: b, size: 2",
    ]


def test_curve_per_x_level_and_one_legend(make_study) -> None:
    results = load_results(load_study(make_study(report=REPORT)))

    figure = plot_study_curves(results)

    assert len(figure.axes) == 2
    assert all(len(axes.get_lines()) == 2 for axes in figure.axes)
    colors = [[line.get_color() for line in axes.get_lines()] for axes in figure.axes]
    assert colors[0] == colors[1]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["sigmoid", "tanh"]
    plt.close(figure)


def test_without_color_one_panel(make_study) -> None:
    results = load_results(load_study(make_study(axes={"activation": ["sigmoid", "tanh"]})))

    assert [panel.title(results.study) for panel in curve_panels(results)] == [None]


def test_axis_fitted_without_first_epochs() -> None:
    figure, axes = plt.subplots()
    # 21 epochs: the first 2 (10 %) are left out.
    climb = np.array([20.0, 60.0, *np.linspace(90.0, 94.0, 19)])

    fit_after_warmup(axes, [climb])

    assert axes.get_ylim() == pytest.approx((90.0 - 0.2, 94.0 + 0.2))
    plt.close(figure)


def test_accuracy_bands_within_bounds(make_study) -> None:
    results = load_results(load_study(make_study(report=REPORT)))

    figure = plot_study_curves(results)

    for axes in figure.axes:
        for band in axes.collections:
            vertices = band.get_paths()[0].vertices
            assert vertices[:, 1].min() >= 0.0
            assert vertices[:, 1].max() <= 100.0
    plt.close(figure)


def test_labels_in_panel_titles_and_legend(make_study) -> None:
    report = REPORT + '\nlabels = { dataset = "Zbiór", a = "A", tanh = "tangens" }'
    results = load_results(load_study(make_study(report=report)))

    figure = plot_study_curves(results)

    assert [axes.get_title() for axes in figure.axes] == ["Zbiór: A", "Zbiór: b"]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["sigmoid", "tangens"]
    plt.close(figure)
