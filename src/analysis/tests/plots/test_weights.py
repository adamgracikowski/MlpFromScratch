import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment import load_experiment
from analysis.plots.weights import (
    _edges,
    _final_values,
    _layer_panels,
    plot_weight_heatmap,
    plot_weight_norms,
    plot_weight_trajectories,
)
from analysis.style import BLUE, VERMILLION


def visible_axes(figure) -> list:
    return [axes for axes in figure.axes if axes.get_visible()]


def test_trajectories_color_biases(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    figure = plot_weight_trajectories(experiment)

    panels = visible_axes(figure)
    assert [axes.get_title() for axes in panels] == ["Warstwa ukryta 1", "Warstwa wyjściowa"]
    for layer, axes in enumerate(panels):
        fan_in = experiment.metadata["network"]["layer_sizes"][layer]
        biases = (experiment.weights.indices[layer][:, 0] == fan_in).sum()
        colors = [line.get_color() for line in axes.get_lines()]
        assert colors.count(VERMILLION) == biases
        assert colors.count(BLUE) == len(experiment.weights.indices[layer]) - biases
    assert len(figure.legends[0].get_texts()) == 2
    plt.close(figure)


def test_trajectories_without_bias(make_experiment) -> None:
    figure = plot_weight_trajectories(load_experiment(make_experiment(bias=False)))

    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["Wagi $w_{ij}$"]
    plt.close(figure)


def test_norms(make_experiment) -> None:
    figure = plot_weight_norms(load_experiment(make_experiment()))
    (axes,) = figure.axes

    assert len(axes.get_lines()) == 4  # ||W|| and ||b|| of 2 layers
    texts = [text.get_text() for text in axes.get_legend().get_texts()]
    assert texts[:2] == ["Warstwa ukryta 1", "Warstwa wyjściowa"]
    plt.close(figure)


def test_norms_without_bias(make_experiment) -> None:
    figure = plot_weight_norms(load_experiment(make_experiment(bias=False)))

    assert len(figure.axes[0].get_lines()) == 2
    plt.close(figure)


def test_heatmap_sorted_by_final_value(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    figure = plot_weight_heatmap(experiment)

    panels = [axes for axes in visible_axes(figure) if axes.get_title()]
    assert len(panels) == 2
    for layer, axes in enumerate(panels):
        image = axes.collections[0].get_array()
        assert image.shape == experiment.weights.values[layer].T.shape
        assert np.all(np.diff(image[:, -1]) >= 0)
    plt.close(figure)


def test_heatmap_after_divergence(make_experiment) -> None:
    experiment = load_experiment(make_experiment(diverged=True))
    experiment.weights.values[0][-1, :] = np.nan

    figure = plot_weight_heatmap(experiment)
    plt.close(figure)


def test_layer_panels_layout() -> None:
    for layer_count, expected_axes in ((1, 1), (3, 3), (4, 4), (5, 6)):
        figure, panels = _layer_panels(layer_count, panel_height=4.0)
        assert len(panels) == layer_count
        assert len(figure.axes) == expected_axes
        assert len(visible_axes(figure)) == layer_count
        plt.close(figure)


def test_final_values_skip_nan() -> None:
    values = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan]])

    np.testing.assert_array_equal(_final_values(values), [3.0, 2.0, np.inf])


def test_edges() -> None:
    np.testing.assert_allclose(_edges(np.array([0.0, 10.0, 20.0, 25.0])), [-5, 5, 15, 22.5, 27.5])
    np.testing.assert_allclose(_edges(np.array([4.0])), [3.5, 4.5])
