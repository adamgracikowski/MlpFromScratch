import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment import load_experiment
from analysis.plots.loss import plot_loss, plot_metrics


def legend_texts(axes) -> list[str]:
    return [text.get_text() for text in axes.get_legend().get_texts()]


def test_loss_panels(make_experiment) -> None:
    figure = plot_loss(load_experiment(make_experiment()))
    by_iteration, by_epoch = figure.axes

    assert by_iteration.get_yscale() == by_epoch.get_yscale() == "log"
    assert "Strata minibatcha" in legend_texts(by_iteration)
    assert "Średnia krocząca (1 epoka)" in legend_texts(by_iteration)
    assert legend_texts(by_epoch) == ["Zbiór uczący", "Zbiór testowy"]
    assert by_epoch.get_ylabel() == "Strata (entropia krzyżowa)"
    plt.close(figure)


def test_best_epoch_marked(make_experiment) -> None:
    experiment = load_experiment(make_experiment(validation=True, early_stopped=True))
    figure = plot_loss(experiment)

    for axes in figure.axes:
        assert "Najlepsza epoka (2)" in legend_texts(axes)
    _, by_epoch = figure.axes
    assert "Zbiór walidacyjny" in legend_texts(by_epoch)
    plt.close(figure)


def test_diverged_run_leaves_gaps(make_experiment) -> None:
    figure = plot_loss(load_experiment(make_experiment(diverged=True)))
    by_iteration, by_epoch = figure.axes

    raw = by_iteration.get_lines()[0].get_ydata()
    assert np.isnan(raw[-2:]).all()
    assert np.isnan(by_epoch.get_lines()[0].get_ydata()[-1])
    plt.close(figure)


def test_metrics_classification_in_percent(make_experiment) -> None:
    figure = plot_metrics(load_experiment(make_experiment()))
    (axes,) = figure.axes

    assert axes.get_ylabel() == "Dokładność [%]"
    np.testing.assert_allclose(axes.get_lines()[0].get_ydata()[[0, -1]], [30.0, 90.0])
    plt.close(figure)


def test_metrics_regression_mse_log(make_experiment) -> None:
    figure = plot_metrics(load_experiment(make_experiment(task="regression", features=1)))
    (axes,) = figure.axes

    assert axes.get_ylabel() == "MSE (skala oryginalna)"
    assert axes.get_yscale() == "log"
    plt.close(figure)
