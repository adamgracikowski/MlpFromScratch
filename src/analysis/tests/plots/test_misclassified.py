import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment import load_experiment
from analysis.plots.misclassified import (
    MAX_IMAGES,
    misclassified_order,
    misclassified_reason,
    plot_misclassified,
)


def test_orders_mistakes_by_confidence() -> None:
    labels = np.array([1, 2, 3, 1, 2])
    predicted = np.array([2, 2, 1, 3, 1])
    outputs = np.array([[0.3, 0.6, 0.1], [0, 1, 0], [0.9, 0, 0.1], [0.4, 0, 0.6], [0.6, 0.4, 0]])

    # Samples 0 and 3 tie at 0.6 and keep their order.
    np.testing.assert_array_equal(misclassified_order(labels, predicted, outputs), [2, 0, 3, 4])


def test_plot_shows_mistakes_with_titles(make_experiment) -> None:
    experiment = load_experiment(make_experiment(features=64))
    test = experiment.predictions["test"]
    mistakes = int(np.sum(test.labels != test.predicted))

    figure = plot_misclassified(experiment)

    images = [axes for axes in figure.axes if axes.images]
    assert len(images) == min(mistakes, MAX_IMAGES)
    assert images[0].images[0].get_array().shape == (8, 8)
    first = misclassified_order(test.labels, test.predicted, test.outputs)[0]
    assert images[0].get_title().startswith(f"{test.labels[first]} jako {test.predicted[first]}\n")
    plt.close(figure)


def test_reason_without_mistakes(make_experiment) -> None:
    experiment = load_experiment(make_experiment(features=64))
    test = experiment.predictions["test"]
    correct = dataclasses.replace(test, predicted=test.labels)
    experiment = dataclasses.replace(experiment, predictions={"test": correct})

    assert misclassified_reason(experiment) == "every test sample is classified correctly"
