from analysis.experiment import load_experiment


def test_loads_trained_parameters(make_experiment) -> None:
    model = load_experiment(make_experiment(features=3)).model

    assert [weights.shape for weights in model.weights] == [(3, 4), (4, 3)]
    assert [bias.shape for bias in model.biases] == [(4,), (3,)]


def test_without_bias(make_experiment) -> None:
    model = load_experiment(make_experiment(bias=False)).model

    assert model.biases == [None, None]
