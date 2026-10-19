import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment import Model, load_experiment
from analysis.plots.first_layer_weights import plot_first_layer_weights


def _maps(figure) -> list:
    return [axes for axes in figure.axes if axes.images]


def test_one_map_per_hidden_neuron(make_experiment) -> None:
    experiment = load_experiment(make_experiment(features=64))

    figure = plot_first_layer_weights(experiment)

    maps = _maps(figure)
    assert len(maps) == experiment.model.weights[0].shape[1]
    assert maps[0].get_title() == "Neuron 1"
    np.testing.assert_allclose(
        maps[1].images[0].get_array().data.ravel(), experiment.model.weights[0][:, 1]
    )
    plt.close(figure)


def test_masks_pixels_constant_in_training_set(make_experiment) -> None:
    experiment = load_experiment(make_experiment(features=64))
    train = experiment.predictions["train"]
    inputs = train.inputs.copy()
    inputs[:, 0] = 0.0
    predictions = experiment.predictions | {"train": dataclasses.replace(train, inputs=inputs)}
    experiment = dataclasses.replace(experiment, predictions=predictions)

    figure = plot_first_layer_weights(experiment)

    mask = np.ma.getmaskarray(_maps(figure)[0].images[0].get_array())
    assert mask[0, 0]
    assert mask.sum() == 1
    plt.close(figure)


def test_output_layer_maps_are_classes(make_experiment) -> None:
    experiment = load_experiment(make_experiment(features=64))
    rng = np.random.default_rng(0)
    single = Model(weights=[rng.normal(size=(64, 3))], biases=[None])

    figure = plot_first_layer_weights(dataclasses.replace(experiment, model=single))

    assert [axes.get_title() for axes in _maps(figure)] == ["Klasa 1", "Klasa 2", "Klasa 3"]
    plt.close(figure)
