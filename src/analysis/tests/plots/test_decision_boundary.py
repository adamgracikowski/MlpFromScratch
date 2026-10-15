import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from analysis.experiment import load_experiment
from analysis.plots.decision_boundary import (
    REGION_ALPHA,
    class_margin,
    plot_decision_boundary,
    region_colors,
)
from analysis.style import BLUE, VERMILLION


def test_region_colors_follow_confidence() -> None:
    probabilities = np.array([[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.2, 0.8]]])

    colors = region_colors(probabilities)

    low, high = REGION_ALPHA
    blue, vermillion = np.array(to_rgb(BLUE)), np.array(to_rgb(VERMILLION))
    np.testing.assert_allclose(colors[0, 0], (1 - high) + high * blue)
    np.testing.assert_allclose(colors[0, 1], (1 - low) + low * blue)  # a tie: argmax = 0
    np.testing.assert_allclose(colors[1, 0], (1 - high) + high * vermillion)
    assert colors.shape == (2, 2, 3)
    assert colors.min() >= 0.0 and colors.max() <= 1.0


def test_class_margin() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]])

    np.testing.assert_allclose(class_margin(probabilities, 0), [0.5, -0.1, 0.0])
    np.testing.assert_allclose(class_margin(probabilities, 2), [-0.6, 0.1, -0.5])


def test_plot(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    figure = plot_decision_boundary(experiment)
    (axes,) = figure.axes

    test = experiment.predictions["test"]
    wrong = int((test.predicted != test.labels).sum())
    scatters = axes.collections[-6:]  # 3 classes, then the misclassified of 3 classes
    assert sum(len(points.get_offsets()) for points in scatters[:3]) == len(test.rows)
    assert sum(len(points.get_offsets()) for points in scatters[3:]) == wrong
    texts = [text.get_text() for text in axes.get_legend().get_texts()]
    assert texts == ["Klasa 1", "Klasa 2", "Klasa 3", "Błędnie sklasyfikowane"]
    assert len(axes.get_images()) == 1
    plt.close(figure)
