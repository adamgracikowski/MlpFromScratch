import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment import load_experiment
from analysis.plots.confusion import confusion_matrix, plot_confusion, row_shares


def test_confusion_matrix_uses_original_labels() -> None:
    counts = confusion_matrix(
        np.array([1, 1, 2, 3, 3, 3]), np.array([1, 3, 2, 3, 3, 1]), classes=[1, 2, 3]
    )

    np.testing.assert_array_equal(counts, [[1, 0, 1], [0, 1, 0], [1, 0, 2]])


def test_row_shares_without_samples() -> None:
    shares = row_shares(np.array([[3, 1], [0, 0]]))

    np.testing.assert_allclose(shares, [[0.75, 0.25], [0.0, 0.0]])


def test_plot(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    figure = plot_confusion(experiment, "test")
    axes = figure.axes[0]

    texts = [text.get_text() for text in axes.texts]
    assert len(texts) == 9
    assert all(text.endswith("%") for text in texts)
    test = experiment.predictions["test"]
    total = sum(int(text.split("\n")[0]) for text in texts)
    assert total == len(test.rows)
    assert axes.get_xlabel() == "Przewidziana klasa"
    plt.close(figure)


def test_plot_ten_classes(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    rng = np.random.default_rng(1)
    test = experiment.predictions["test"]
    labels = rng.integers(0, 10, size=len(test.rows))
    predicted = rng.integers(0, 10, size=len(test.rows))
    experiment = dataclasses.replace(
        experiment,
        metadata={
            **experiment.metadata,
            "data": {**experiment.metadata["data"], "classes": list(range(10))},
        },
        predictions={"test": dataclasses.replace(test, labels=labels, predicted=predicted)},
    )

    figure = plot_confusion(experiment, "test")

    assert len(figure.axes[0].texts) == 100
    plt.close(figure)
