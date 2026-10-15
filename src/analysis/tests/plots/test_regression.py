import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment import load_experiment
from analysis.plots.regression import plot_regression


def legend_texts(axes) -> list[str]:
    return [text.get_text() for text in axes.get_legend().get_texts()]


def test_panels(make_experiment) -> None:
    experiment = load_experiment(make_experiment(task="regression", features=1))
    figure = plot_regression(experiment)
    result, residuals = figure.axes

    assert legend_texts(result)[:3] == [
        "Funkcja docelowa (zbiór testowy)",
        "Zbiór uczący",
        "Predykcja sieci",
    ]
    test = experiment.predictions["test"]
    order = np.argsort(test.inputs[:, 0], kind="stable")
    line = residuals.get_lines()[-1]
    np.testing.assert_allclose(line.get_xdata(), test.inputs[order, 0])
    np.testing.assert_allclose(line.get_ydata(), (test.outputs[:, 0] - test.targets)[order])
    assert residuals.get_ylabel() == r"Reszta $\hat{y} - y$"
    plt.close(figure)


def test_marks_extrapolation(make_experiment) -> None:
    # The grid spans -3..3, wider than the training inputs drawn from N(0, 1).
    figure = plot_regression(load_experiment(make_experiment(task="regression", features=1)))
    result, _ = figure.axes

    assert "Poza zakresem zbioru uczącego" in legend_texts(result)
    plt.close(figure)


def test_validation_points(make_experiment) -> None:
    experiment_dir = make_experiment(task="regression", features=1, validation=True)
    figure = plot_regression(load_experiment(experiment_dir))
    result, _ = figure.axes

    assert "Zbiór walidacyjny" in legend_texts(result)
    plt.close(figure)
