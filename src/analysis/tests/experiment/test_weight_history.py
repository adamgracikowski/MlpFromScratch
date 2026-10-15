import numpy as np

from analysis.experiment import WeightHistory


def test_loads_snapshots(make_experiment) -> None:
    weights = WeightHistory.load(make_experiment() / "artifacts/weights", layer_count=2)

    snapshots = len(weights.iteration)
    assert weights.iteration[0] == 0
    assert weights.layer_count == 2
    assert weights.weight_norms.shape == weights.bias_norms.shape == (snapshots, 2)
    for indices, values in zip(weights.indices, weights.values, strict=True):
        assert indices.shape[1] == 2
        assert values.shape == (snapshots, len(indices))


def test_bias_norms_nan_without_bias(make_experiment) -> None:
    weights = WeightHistory.load(make_experiment(bias=False) / "artifacts/weights", layer_count=2)

    assert np.isnan(weights.bias_norms).all()
    assert np.isfinite(weights.weight_norms).all()
